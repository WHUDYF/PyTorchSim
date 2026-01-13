import math
import torch
import warnings

from torch._inductor.codegen import common
from torch._inductor.virtualized import V, _ops as ops
from . import mlir_common

warnings.filterwarnings('ignore', message='undefined OpHandler\\..*, please add missing op schema')

def reduction_combine_vec(reduction_type, vector_value, init_value, axis, shape, reduced_shape):
    if reduction_type == "sum":
        return f"vector.multi_reduction <add>, %{vector_value}, %{init_value} [{axis}] : {shape} to {reduced_shape}"
    if reduction_type == "prod":
        return f"vector.multi_reduction <mul>, %{vector_value}, %{init_value} [{axis}] : {shape} to {reduced_shape}"
    if reduction_type == "max":
        return f"vector.multi_reduction <maximumf>, %{vector_value}, %{init_value} [{axis}] : {shape} to {reduced_shape}"
    if reduction_type == "min":
        return f"vector.multi_reduction <minimumf>, %{vector_value}, %{init_value} [{axis}] : {shape} to {reduced_shape}"
    if reduction_type == "any":
        return f"vector.multi_reduction <or>, %{vector_value}, %{init_value} [{axis}] : {shape} to {reduced_shape}"
    raise AssertionError(reduction_type)

class ExtensionOverrides(common.OpOverrides):
    @staticmethod
    def constant(value, src_type, *args, **kwargs):
        if isinstance(src_type, torch.dtype):
            src_type = mlir_common.DTYPE_TO_MLIR[src_type]

        str_val = str(value)
        if "inf" == str_val or "-inf" == str_val or "nan" == str_val:
            value = f"0x{mlir_common.MLIR_INF[str_val][src_type]:x}"
        # scientific notation check
        elif "e" in str_val:
            value = format(float(value), ".20f")
        elif src_type[0] == "f":
            value = format(float(value), ".20f")
        elif src_type[0] == "i":
            value = int(float(value)) 
        return f'arith.constant {value} : {src_type}', [1, src_type]

    @staticmethod
    def broadcast(operand, target_size, *args, **kwargs):
        src_size, dtype = V.kernel.var_info[operand]

        src_shape = f"vector<{src_size}x{dtype}>" if src_size > 1 else dtype
        dst_shape = f"vector<{target_size}x{dtype}>"

        op_str = ""
        # Special case for length 2 vector. We used this vector to avoid scalar operations...
        if src_size > 1:
            if target_size % src_size == 0:
                unflat_operand = ops.broadcast_unflat(operand, target_size)
                outer_dim = target_size // src_size
                unflat_shape = f"vector<{outer_dim}x{src_size}x{dtype}>"
                # Flatten back to 1D
                op_str = f"vector.shape_cast %{unflat_operand} : {unflat_shape} to {dst_shape}"
            else:
                raise NotImplementedError(
                    f"Vector broadcast size mismatch: src={src_size} cannot broadcast to target={target_size}"
                )
        elif src_size == 1:
            op_str = f"vector.broadcast %{operand} : {src_shape} to {dst_shape}"
        else:
            raise ValueError(f"Invalid source size: {src_size}")
        return op_str, [target_size, dtype]

    @staticmethod
    def broadcast_unflat(operand, target_size, *args, **kwargs):
        src_size, dtype = V.kernel.var_info[operand]

        outer_dim = target_size // src_size
        src_shape = f"vector<{src_size}x{dtype}>"
        dst_shape = f"vector<{outer_dim}x{src_size}x{dtype}>"

        op_str = f"vector.broadcast %{operand} : {src_shape} to {dst_shape}"
        return op_str, [target_size, dtype]

    def load_seed(self, *args, **kwargs):
        raise NotImplementedError

    def rand(self, *args, **kwargs):
        raise NotImplementedError

    def randn(self, *args, **kwargs):
        raise NotImplementedError

    def randint64(self, *args, **kwargs):
        raise NotImplementedError

    # Special operaitons
    @staticmethod
    def masked(mask, body, other, *args, tile_size=16, dtype="f32", ninf_declared=False, **kwargs):
        result = body()
        val = ops.constant(other, dtype, *args, **kwargs)
        result = ops.where(mask, result, val)
        return result, V.kernel.var_info[result]

    @staticmethod
    def where(condition, operand1, operand2, *args, **kwargs):
        tile_size, ret_type, operand1, operand2 = ExtensionOverrides.binary_elementwise_common(operand1, operand2)
        cond_type = V.kernel.var_info[condition]
        operand_type = V.kernel.var_info[operand1]
        condition = ops.to_bool(condition)
        if cond_type[0] < tile_size:
            condition = ops.broadcast(condition, tile_size)
        elif cond_type[0] > tile_size:
            operand1 = ops.broadcast(operand1, cond_type[0])
            operand2 = ops.broadcast(operand2, cond_type[0])
        tile_size, ret_type = V.kernel.var_info[operand1]
        shape = f"vector<{tile_size}x{ret_type}>" if tile_size > 1 else ret_type
        cond_shape = f"vector<{tile_size}xi1>" if tile_size > 1 else ""
        return f"arith.select %{condition}, %{operand1}, %{operand2} : {cond_shape}, {shape}", [tile_size, ret_type]

    @staticmethod
    def to_dtype(operand, dst_mlir_dtype, *args, **kwargs):
        # Extract source information
        src_mlir_dtype = V.kernel.var_info[operand][1]
        tile_size = V.kernel.var_info[operand][0]

        # Normalize destination type (Torch dtype -> MLIR string)
        if isinstance(dst_mlir_dtype, torch.dtype):
            dst_mlir_dtype = mlir_common.DTYPE_TO_MLIR[dst_mlir_dtype]

        if src_mlir_dtype == "index" and dst_mlir_dtype != "index":
            operand = ops.index_cast(operand, "i64")
            src_mlir_dtype = "i64" # Update explicitly

        if dst_mlir_dtype == "index":
            # If source is already index, return as is; otherwise cast
            if src_mlir_dtype == "index":
                return operand, [tile_size, "index"]
            return ops.index_cast(operand, "index"), [tile_size, "index"]

        # Early return if types are identical
        if src_mlir_dtype == dst_mlir_dtype:
            return operand, [tile_size, dst_mlir_dtype]

        dst_bits = mlir_common.MLIR_TO_BIT[dst_mlir_dtype]
        src_bits = mlir_common.MLIR_TO_BIT[src_mlir_dtype]
        shape = f"vector<{tile_size}x{dst_mlir_dtype}>" if tile_size > 1 else dst_mlir_dtype
        src_shape = f"vector<{tile_size}x{src_mlir_dtype}>" if tile_size > 1 else src_mlir_dtype
        src_type_char = src_mlir_dtype[0] # 'i' or 'f'
        dst_type_char = dst_mlir_dtype[0] # 'i' or 'f'o

        op_str = ""

        # Case A: Integer -> Float
        if src_type_char == "i" and dst_type_char == "f":
            op_str = f"arith.sitofp %{operand} : {src_shape} to {shape}"
        # Case B: Float -> Integer
        elif src_type_char == "f" and dst_type_char == "i":
            op_str = f"arith.fptosi %{operand} : {src_shape} to {shape}"
        # Case C: Integer -> Integer (Extension / Truncation)
        elif src_type_char == "i" and dst_type_char == "i":
            if dst_bits > src_bits:
                op_str = f"arith.extsi %{operand} : {src_shape} to {shape}"
            elif dst_bits < src_bits:
                # Use arith.trunci for integer truncation
                op_str = f"arith.trunci %{operand} : {src_shape} to {shape}" 
            else:
                return operand, [tile_size, dst_mlir_dtype]
        # Case D: Float -> Float (Extension / Truncation)
        elif src_type_char == "f" and dst_type_char == "f":
            if dst_bits > src_bits:
                op_str = f"arith.extf %{operand} : {src_shape} to {shape}"
            elif dst_bits < src_bits:
                # Corrected 'trunf' to 'truncf'
                op_str = f"arith.truncf %{operand} : {src_shape} to {shape}" 
            else:
                return operand, [tile_size, dst_mlir_dtype]
        else:
            raise NotImplementedError(f"Unsupported conversion: {src_mlir_dtype} -> {dst_mlir_dtype}")

        return op_str, [tile_size, dst_mlir_dtype]

    @staticmethod
    def identity(operand, *args, **kwargs):
        operand_info = V.kernel.var_info[operand]
        return operand, operand_info

    @staticmethod
    def to_dtype_bitcast(operand, dtype, *args, **kwargs):
        tile_size, current_src_type = V.kernel.var_info[operand]

        if isinstance(dtype, torch.dtype):
            dst_mlir_type = mlir_common.DTYPE_TO_MLIR[dtype]
        else:
            dst_mlir_type = dtype

        src_bits = mlir_common.MLIR_TO_BIT[current_src_type]
        dst_bits = mlir_common.MLIR_TO_BIT[dst_mlir_type]

        if src_bits != dst_bits:
            raise ValueError(
                f"Bitcast failed: Bit width mismatch. "
                f"Src: {current_src_type}({src_bits}b) != Dst: {dst_mlir_type}({dst_bits}b)"
            )

        src_shape = f"vector<{tile_size}x{current_src_type}>" if tile_size > 1 else current_src_type
        dst_shape = f"vector<{tile_size}x{dst_mlir_type}>" if tile_size > 1 else dst_mlir_type

        return f"arith.bitcast %{operand} : {src_shape} to {dst_shape}", [tile_size, dst_mlir_type]

    # Binary element wise operations
    @staticmethod
    def binary_elementwise_common(operand1, operand2):
        V.kernel.var_info = V.kernel.var_info
        operand1.bounds = operand1.bounds.unknown()
        operand2.bounds = operand2.bounds.unknown()
        op_type1 = V.kernel.var_info[operand1]
        op_type2 = V.kernel.var_info[operand2]
        # Tile size check
        if op_type1[0] != op_type2[0]:
            # Try to broad cast
            lhs_tile_size, lhs_dtype = op_type1
            rhs_tile_size, rhs_dtype = op_type2
            if lhs_tile_size > rhs_tile_size:
                operand2 = ops.broadcast(operand2, lhs_tile_size)
                op_type2 = V.kernel.var_info[operand2]
            elif lhs_tile_size < rhs_tile_size:
                operand1 = ops.broadcast(operand1, rhs_tile_size)
                op_type1 = V.kernel.var_info[operand1]

        # Data type check
        if op_type1[1] != op_type2[1]:
            if op_type1[1] == "index" or op_type1 == "index":
                if op_type1[1] == "index":
                    # index -> target type: 2-step casting if target is float
                    if op_type2[1][0] == "f":
                        operand1 = ops.index_cast(operand1, "i64")
                        operand1 = ops.to_dtype(operand1, op_type2[1])
                        op_type1 = V.kernel.var_info[operand1]
                    else:
                        # index -> integer: direct casting
                        operand1 = ops.index_cast(operand1, op_type2[1])
                        op_type1 = V.kernel.var_info[operand1]
                if op_type2[1] == "index":
                    # index -> target type: 2-step casting if target is float
                    if op_type1[1][0] == "f":
                        operand2 = ops.index_cast(operand2, "i64")
                        operand2 = ops.to_dtype(operand2, op_type1[1])
                        op_type2 = V.kernel.var_info[operand2]
                    else:
                        # index -> integer: direct casting
                        operand2 = ops.index_cast(operand2, op_type1[1])
                        op_type2 = V.kernel.var_info[operand2]
            elif op_type1[1][0] == "i" and op_type2[1][0] == "f":
                operand1 = ops.to_dtype(operand1, op_type2[1])
                op_type1 = V.kernel.var_info[operand1]
            elif op_type1[1][0] == "f" and op_type2[1][0] == "i":
                operand2 = ops.to_dtype(operand2, op_type1[1])
                op_type2 = V.kernel.var_info[operand2]
            elif op_type1[1][0] == op_type2[1][0]:
                if mlir_common.MLIR_TO_BIT[op_type1[1]] > mlir_common.MLIR_TO_BIT[op_type2[1]]:
                   operand2 = ops.ext(operand2, op_type1[1])
                   op_type2 = V.kernel.var_info[operand2]
                elif mlir_common.MLIR_TO_BIT[op_type1[1]] < mlir_common.MLIR_TO_BIT[op_type2[1]]:
                   operand1 = ops.ext(operand1, op_type2[1])
                   op_type1 = V.kernel.var_info[operand1]
            else:
                raise NotImplementedError("Unsupported type converting")

        # Updated var info
        tile_size = op_type1[0]
        ret_type = op_type1[1]
        return tile_size, ret_type, operand1, operand2

    @staticmethod
    def abs(operand, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def exp(operand, *args, **kwargs):
        # Check scalar
        op_type = V.kernel.var_info[operand]
        if op_type[0] == 1:
            operand = ops.broadcast(operand, 4)
            val = ops.exp(operand)
            result = ops.extractelement(val, 0)
            return result, V.kernel.var_info[result]
        op_type = V.kernel.var_info[operand]
        tile_size = op_type[0]
        dtype = op_type[1]
        shape = f"vector<{tile_size}x{dtype}>" if tile_size > 1 else dtype
        return f'math.exp %{operand} : {shape}', [tile_size, dtype]

    @staticmethod
    def exp2(operand, *args, **kwargs):
        # Hands-on part: implement exp2 using math.exp2
        # V.kernel.var_info = {operand: [tile_size, dtype]}
        # Ex) V.kernel.var_info[operand] = [8, "f32"]

        ln2 = math.log(2)
        coeff = ops.constant(ln2, "f32")
        operand = ops.mul(operand, coeff)
        return ops.exp(operand), V.kernel.var_info[operand]

    @staticmethod
    def expm1(operand, *args, **kwargs):
        coeff = ops.constant(1.0, "f32")
        operand = ops.exp(operand)
        operand = ops.sub(operand, coeff)
        return operand, V.kernel.var_info[operand]

    @staticmethod
    def sqrt(operand, *args, **kwargs):
        op_type = V.kernel.var_info[operand]

        tile_size = op_type[0]
        dtype = op_type[1]

        # Type check & auto cast
        if dtype.startswith("f"):
            operand = ops.to_dtype(operand, "f32")

        shape = f"vector<{tile_size}x{dtype}>" if tile_size > 1 else dtype
        return f'math.sqrt %{operand} : {shape}', [tile_size, dtype]

    @staticmethod
    def relu(operand, *args, **kwargs):
        src_mlir_dtype = V.kernel.var_info[operand][1]
        tile_size = V.kernel.var_info[operand][0]
        return ops.maximum(operand, ops.constant(0, src_mlir_dtype)), [tile_size, src_mlir_dtype]

    @staticmethod
    def minimum(operand1, operand2, *args, **kwargs):
        tile_size, ret_type, operand1, operand2 = ExtensionOverrides.binary_elementwise_common(operand1, operand2)
        shape = f"vector<{tile_size}x{ret_type}>" if tile_size > 1 else ret_type
        if ret_type[0] == "f":
            opcode = f'arith.minimumf'
        else:
            opcode = f'arith.minsi'
        return f'{opcode} %{operand1}, %{operand2} : {shape}', [tile_size, ret_type]

    @staticmethod
    def maximum(operand1, operand2, *args, **kwargs):
        tile_size, ret_type, operand1, operand2 = ExtensionOverrides.binary_elementwise_common(operand1, operand2)
        shape = f"vector<{tile_size}x{ret_type}>" if tile_size > 1 else ret_type
        if ret_type[0] == "f":
            opcode = f'arith.maximumf'
        else:
            opcode = f'arith.maxsi'
        return f'{opcode} %{operand1}, %{operand2} : {shape}', [tile_size, ret_type]

    @staticmethod
    def cos(operand, *args, **kwargs):
        op_type = V.kernel.var_info[operand]

        # Check scalar
        op_type = V.kernel.var_info[operand]
        if op_type[0] == 1:
            operand = ops.broadcast(operand, 4)
            val = ops.cos(operand)
            result = ops.extractelement(val, 0)
            return result, V.kernel.var_info[result]
        op_type = V.kernel.var_info[operand]
        tile_size = op_type[0]
        dtype = op_type[1]

        # Type check & auto cast
        if dtype.startswith("f"):
            operand = ops.to_dtype(operand, "f32")
        shape = f"vector<{tile_size}x{dtype}>" if tile_size > 1 else dtype
        return f'math.cos %{operand} : {shape}', [tile_size, dtype]

    @staticmethod
    def sin(operand, *args, **kwargs):
        op_type = V.kernel.var_info[operand]

        # Check scalar
        op_type = V.kernel.var_info[operand]
        if op_type[0] == 1:
            operand = ops.broadcast(operand, 4)
            val = ops.sin(operand)
            result = ops.extractelement(val, 0)
            return result, V.kernel.var_info[result]
        op_type = V.kernel.var_info[operand]
        tile_size = op_type[0]
        dtype = op_type[1]

        # Type check & auto cast
        if dtype.startswith("f"):
            operand = ops.to_dtype(operand, "f32")
        shape = f"vector<{tile_size}x{dtype}>" if tile_size > 1 else dtype
        return f'math.sin %{operand} : {shape}', [tile_size, dtype]

    @staticmethod
    def tan(operand, *args, **kwargs):
        sin_res = ops.sin(operand)
        cos_res = ops.cos(operand)
        operand = ops.truediv(sin_res, cos_res)
        return operand, V.kernel.var_info[operand]

    @staticmethod
    def lgamma(operand, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def erf(operand, *args, **kwargs):
        # Check scalar
        op_type = V.kernel.var_info[operand]
        if op_type[0] == 1:
            operand = ops.broadcast(operand, 4)
            val = ops.erf(operand)
            result = ops.extractelement(val, 0)
            return result, V.kernel.var_info[result]
        op_type = V.kernel.var_info[operand]
        tile_size = op_type[0]
        dtype = op_type[1]
        shape = f"vector<{tile_size}x{dtype}>" if tile_size > 1 else dtype
        return f'math.erf %{operand} : {shape}', [tile_size, dtype]

    @staticmethod
    def cosh(operand, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def sinh(operand, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def tanh(operand, *args, **kwargs):
        op_type = V.kernel.var_info[operand]

        # Check scalar
        op_type = V.kernel.var_info[operand]
        if op_type[0] == 1:
            operand = ops.broadcast(operand, 4)
            val = ops.tanh(operand)
            result = ops.extractelement(val, 0)
            return result, V.kernel.var_info[result]
        op_type = V.kernel.var_info[operand]
        tile_size = op_type[0]
        dtype = op_type[1]

        # Type check & auto cast
        if dtype.startswith("f"):
            operand = ops.to_dtype(operand, "f32")
        shape = f"vector<{tile_size}x{dtype}>" if tile_size > 1 else dtype
        return f'math.tanh %{operand} : {shape}', [tile_size, dtype]

    @staticmethod
    def acos(operand, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def acosh(operand, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def asin(operand, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def asinh(operand, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def atan2(operand1, operand2, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def atan(operand, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def atanh(operand, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def copysign(operand1, operand2, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def erfc(operand, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def erfinv(operand, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def frexp(operand, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def hypot(operand1, operand2, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def log10(operand, *args, **kwargs):
        val_ln = ops.log(operand)
        
        tile_size, dtype = V.kernel.var_info[val_ln]
        inv_ln10 = 1/math.log(10)
        const_op = ops.constant(inv_ln10, dtype)
        
        # Multiply: ln(x) * (1/ln(10))
        result = ops.mul(val_ln, const_op)
        return result, V.kernel.var_info[result]

    @staticmethod
    def log2(operand, *args, **kwargs):
        val_ln = ops.log(operand)
        
        tile_size, dtype = V.kernel.var_info[val_ln]
        inv_ln10 = 1/math.log(2)
        const_op = ops.constant(inv_ln10, dtype)
        
        # Multiply: ln(x) * (1/ln(10))
        result = ops.mul(val_ln, const_op)
        return result, V.kernel.var_info[result]

    @staticmethod
    def log(operand, *args, **kwargs):
        op_type = V.kernel.var_info[operand]
        tile_size = op_type[0]
        dtype = op_type[1]

        # Type check & auto cast
        if dtype.startswith("f"):
            operand = ops.to_dtype(operand, "f32")

        shape = f"vector<{tile_size}x{dtype}>" if tile_size > 1 else dtype
        return f'math.log %{operand} : {shape}', [tile_size, dtype]

    @staticmethod
    def log1p(operand, *args, **kwargs):
        tile_size, dtype = V.kernel.var_info[operand]
        const_one = ops.constant(1, dtype)

        val_add = ops.add(operand, const_one)
        result = ops.log(val_add)
        return result, V.kernel.var_info[result]

    @staticmethod
    def nextafter(operand1, operand2, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def logical_and(operand1, operand2, *args, **kwargs):
        if V.kernel.var_info[operand1][1] != "i1":
            operand1 = ops.to_bool(operand1)
        
        if V.kernel.var_info[operand2][1] != "i1":
            operand2 = ops.to_bool(operand2)
        result = ops.and_(operand1, operand2)
        return result, V.kernel.var_info[result]

    @staticmethod
    def logical_or(operand1, operand2, *args, **kwargs):
        if V.kernel.var_info[operand1][1] != "i1":
            operand1 = ops.to_bool(operand1)
        
        if V.kernel.var_info[operand2][1] != "i1":
            operand2 = ops.to_bool(operand2)
        result = ops.or_(operand1, operand2)
        return result, V.kernel.var_info[result]

    @staticmethod
    def logical_xor(operand1, operand2, *args, **kwargs):
        if V.kernel.var_info[operand1][1] != "i1":
            operand1 = ops.to_bool(operand1)
        
        if V.kernel.var_info[operand2][1] != "i1":
            operand2 = ops.to_bool(operand2)
        result = ops.xor(operand1, operand2)
        return result, V.kernel.var_info[result]
    
    @staticmethod
    def logical_not(operand, *args, **kwargs):
        op_info = V.kernel.var_info[operand]
        tile_size = op_info[0]
        dtype = op_info[1]
        
        zero_const = ops.constant(0, dtype)
        result = ops.eq(operand, zero_const)
        return result, V.kernel.var_info[result]

    @staticmethod
    def bitwise_and(operand1, operand2, *args, **kwargs):
        # Float check
        if V.kernel.var_info[operand1][1].startswith("f") or V.kernel.var_info[operand2][1].startswith("f"):
            raise ValueError("Bitwise AND not supported for floats")
            
        result = ops.and_(operand1, operand2)
        return result, V.kernel.var_info[result]

    @staticmethod
    def bitwise_not(operand, *args, **kwargs):
        tile_size, dtype = V.kernel.var_info[operand]
        # Float check
        if V.kernel.var_info[operand][1].startswith("f"):
            raise ValueError("Bitwise NOT not supported for floats")
        
        neg_one = ops.constant(-1, dtype)
        result = ops.xor(operand, neg_one) 
        return result, V.kernel.var_info[result]

    @staticmethod
    def bitwise_or(operand1, operand2, *args, **kwargs):
        # Float check
        if V.kernel.var_info[operand1][1].startswith("f") or V.kernel.var_info[operand2][1].startswith("f"):
            raise ValueError("Bitwise AND not supported for floats")
            
        result = ops.or_(operand1, operand2)
        return result, V.kernel.var_info[result]

    @staticmethod
    def bitwise_xor(operand1, operand2, *args, **kwargs):
                # Float check
        if V.kernel.var_info[operand1][1].startswith("f") or V.kernel.var_info[operand2][1].startswith("f"):
            raise ValueError("Bitwise AND not supported for floats")
            
        result = ops.xor(operand1, operand2)
        return result, V.kernel.var_info[result]

    @staticmethod
    def bitwise_left_shift(operand1, operand2, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def bitwise_right_shift(operand1, operand2, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def rsqrt(operand, *args, **kwargs):
        op_type = V.kernel.var_info[operand]
        tile_size = op_type[0]
        dtype = op_type[1]

        # Type check & auto cast
        if dtype.startswith("f"):
            operand = ops.to_dtype(operand, "f32")

        shape = f"vector<{tile_size}x{dtype}>" if tile_size > 1 else dtype
        return f'math.rsqrt %{operand} : {shape}', [tile_size, dtype]

    @staticmethod
    def sigmoid(operand, *args, **kwargs):
        op_type = V.kernel.var_info[operand]
        tile_size = op_type[0]
        dtype = op_type[1]
        one = ops.constant(1, dtype)
        return ops.truediv(one, ops.add(one, ops.exp(ops.neg(operand)))), [tile_size, dtype]

    @staticmethod
    def fmod(operand1, operand2, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def isinf(operand, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def isnan(operand, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def round(operand, *args, **kwargs):
        tile_size, dtype = V.kernel.var_info[operand]
        shape = f"vector<{tile_size}x{dtype}>" if tile_size > 1 else dtype

        if dtype.startswith("f"):
            return f"math.roundeven %{operand} : {shape}", [tile_size, dtype]
        else:
            return operand, [tile_size, dtype]

    @staticmethod
    def floor(operand, *args, **kwargs):
        tile_size, dtype = V.kernel.var_info[operand]
        shape = f"vector<{tile_size}x{dtype}>" if tile_size > 1 else dtype

        if dtype.startswith("f"):
            return f"math.floor %{operand} : {shape}", [tile_size, dtype]
        else:
            return operand, [tile_size, dtype]

    @staticmethod
    def sign(operand, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def trunc(operand, *args, **kwargs):
        tile_size, dtype = V.kernel.var_info[operand]
        shape = f"vector<{tile_size}x{dtype}>" if tile_size > 1 else dtype

        if dtype.startswith("f"):
            return f"math.trunc %{operand} : {shape}", [tile_size, dtype]
        else:
            return operand, [tile_size, dtype]

    @staticmethod
    def ceil(operand, *args, **kwargs):
        tile_size, dtype = V.kernel.var_info[operand]
        shape = f"vector<{tile_size}x{dtype}>" if tile_size > 1 else dtype

        if dtype.startswith("f"):
            return f"math.ceil %{operand} : {shape}", [tile_size, dtype]
        else:
            return operand, [tile_size, dtype]

    # Logical operations
    @staticmethod
    def neg(operand, *args, **kwargs):
        op_type = V.kernel.var_info[operand]
        tile_size = op_type[0]
        dtype = op_type[1]

        # Type check & auto cast
        if dtype.startswith("f"):
            operand = ops.to_dtype(operand, "f32")

        shape = f"vector<{tile_size}x{dtype}>" if tile_size > 1 else dtype
        return f'arith.negf %{operand} : {shape}', [tile_size, dtype]

    @staticmethod
    def reciprocal(operand, *args, **kwargs):
        op_type = V.kernel.var_info[operand]
        tile_size = op_type[0]
        dtype = op_type[1]

        # Type check & auto cast
        if dtype.startswith("f"):
            operand = ops.to_dtype(operand, "f32")

        return ops.truediv(ops.constant(1.0, dtype), operand), [tile_size, dtype]

    @staticmethod
    def eq(operand1, operand2, *args, **kwargs):
        tile_size, ret_type, operand1, operand2 = ExtensionOverrides.binary_elementwise_common(operand1, operand2)
        if ret_type[0] == "f":
            op_type = "arith.cmpf"
            attribute = "oeq"
        elif ret_type[0] == "i":
            op_type = "arith.cmpi"
            attribute = "eq"
        else:
            raise ValueError(f"Unsupported data type for 'eq' operation: {ret_type}")

        shape = f"vector<{tile_size}x{ret_type}>" if tile_size > 1 else ret_type
        return f'{op_type} {attribute}, %{operand1}, %{operand2} : {shape}', [tile_size, "i1"]

    @staticmethod
    def ne(operand1, operand2, *args, **kwargs):
        tile_size, ret_type, operand1, operand2 = ExtensionOverrides.binary_elementwise_common(operand1, operand2)
        if ret_type[0] == "f":
            op_type = "arith.cmpf"
            attribute = "one"
        elif ret_type[0] == "i":
            op_type = "arith.cmpi"
            attribute = "ne"
        else:
            raise ValueError(f"Unsupported data type for 'ne' operation: {ret_type}")

        shape = f"vector<{tile_size}x{ret_type}>" if tile_size > 1 else ret_type
        return f'{op_type} {attribute}, %{operand1}, %{operand2} : {shape}', [tile_size, "i1"]

    @staticmethod
    def lt(operand1, operand2, *args, **kwargs):
        tile_size, ret_type, operand1, operand2 = ExtensionOverrides.binary_elementwise_common(operand1, operand2)
        if ret_type[0] == "f":
            op_type = "arith.cmpf"
            attribute = "olt"
        elif ret_type[0] == "i":
            op_type = "arith.cmpi"
            attribute = "slt"
        else:
            raise ValueError(f"Unsupported data type for 'lt' operation: {ret_type}")

        shape = f"vector<{tile_size}x{ret_type}>" if tile_size > 1 else ret_type
        return f'{op_type} {attribute}, %{operand1}, %{operand2} : {shape}', [tile_size, "i1"]

    @staticmethod
    def gt(operand1, operand2, *args, **kwargs):
        tile_size, ret_type, operand1, operand2 = ExtensionOverrides.binary_elementwise_common(operand1, operand2)
        if ret_type[0] == "f":
            op_type = "arith.cmpf"
            attribute = "ogt"
        elif ret_type[0] == "i":
            op_type = "arith.cmpi"
            attribute = "sgt"
        else:
            raise ValueError(f"Unsupported data type for 'gt' operation: {ret_type}")

        shape = f"vector<{tile_size}x{ret_type}>" if tile_size > 1 else ret_type
        return f'{op_type} {attribute}, %{operand1}, %{operand2} : {shape}', [tile_size, "i1"]

    @staticmethod
    def le(operand1, operand2, *args, **kwargs):
        tile_size, ret_type, operand1, operand2 = ExtensionOverrides.binary_elementwise_common(operand1, operand2)
        if ret_type[0] == "f":
            op_type = "arith.cmpf"
            attribute = "ole"
        elif ret_type[0] == "i":
            op_type = "arith.cmpi"
            attribute = "sle"
        else:
            raise ValueError(f"Unsupported data type for 'le' operation: {ret_type}")

        shape = f"vector<{tile_size}x{ret_type}>" if tile_size > 1 else ret_type
        return f'{op_type} {attribute}, %{operand1}, %{operand2} : {shape}', [tile_size, "i1"]

    @staticmethod
    def ge(operand1, operand2, *args, **kwargs):
        tile_size, ret_type, operand1, operand2 = ExtensionOverrides.binary_elementwise_common(operand1, operand2)
        if ret_type[0] == "f":
            op_type = "arith.cmpf"
            attribute = "oge"
        elif ret_type[0] == "i":
            op_type = "arith.cmpi"
            attribute = "sge"
        else:
            raise ValueError(f"Unsupported data type for 'ne' operation: {ret_type}")

        shape = f"vector<{tile_size}x{ret_type}>" if tile_size > 1 else ret_type
        return f'{op_type} {attribute}, %{operand1}, %{operand2} : {shape}', [tile_size, "i1"]

    @staticmethod
    def add(operand1, operand2, *args, **kwargs):
        tile_size, ret_type, operand1, operand2 = ExtensionOverrides.binary_elementwise_common(operand1, operand2)
        shape = f"vector<{tile_size}x{ret_type}>" if tile_size > 1 else ret_type
        opcode = f'arith.add{ret_type[0]}'
        return f'{opcode} %{operand1}, %{operand2} : {shape}', [tile_size, ret_type]

    @staticmethod
    def sub(operand1, operand2, *args, **kwargs):
        tile_size, ret_type, operand1, operand2 = ExtensionOverrides.binary_elementwise_common(operand1, operand2)
        shape = f"vector<{tile_size}x{ret_type}>" if tile_size > 1 else ret_type
        opcode = f'arith.sub{ret_type[0]}'
        return f'{opcode} %{operand1}, %{operand2} : {shape}', [tile_size, ret_type]

    @staticmethod
    def mul(operand1, operand2, *args, **kwargs):
        tile_size, ret_type, operand1, operand2 = ExtensionOverrides.binary_elementwise_common(operand1, operand2)
        shape = f"vector<{tile_size}x{ret_type}>" if tile_size > 1 else ret_type
        opcode = f'arith.mul{ret_type[0]}'
        return f'{opcode} %{operand1}, %{operand2} : {shape}', [tile_size, ret_type]

    @staticmethod
    def pow(operand1, operand2, *args, **kwargs):
        tile_size, ret_type, operand1, operand2 = ExtensionOverrides.binary_elementwise_common(operand1, operand2)
        # Type check & auto cast
        if ret_type.startswith("f"):
            operand1 = ops.to_dtype(operand1, "f32")

        # Type check & auto cast
        if ret_type.startswith("f"):
            operand2 = ops.to_dtype(operand2, "f32")

        shape = f"vector<{tile_size}x{ret_type}>" if tile_size > 1 else ret_type
        return f"math.pow{ret_type[0]} %{operand1}, %{operand2} : {shape}", [tile_size, ret_type]

    @staticmethod
    def and_(operand1, operand2, *args, **kwargs):
        tile_size, ret_type, operand1, operand2 = ExtensionOverrides.binary_elementwise_common(operand1, operand2)
        
        shape = f"vector<{tile_size}x{ret_type}>" if tile_size > 1 else ret_type
        return f'arith.andi %{operand1}, %{operand2} : {shape}', [tile_size, ret_type]

    @staticmethod
    def or_(operand1, operand2, *args, **kwargs):
        tile_size, ret_type, operand1, operand2 = ExtensionOverrides.binary_elementwise_common(operand1, operand2)
        
        shape = f"vector<{tile_size}x{ret_type}>" if tile_size > 1 else ret_type
        return f'arith.ori %{operand1}, %{operand2} : {shape}', [tile_size, ret_type]

    @staticmethod
    def xor(operand1, operand2, *args, **kwargs):
        tile_size, ret_type, operand1, operand2 = ExtensionOverrides.binary_elementwise_common(operand1, operand2)
        
        shape = f"vector<{tile_size}x{ret_type}>" if tile_size > 1 else ret_type
        return f'arith.xori %{operand1}, %{operand2} : {shape}', [tile_size, ret_type]

    @staticmethod
    def lshift(operand1, operand2, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def rshift(operand1, operand2, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def truncdiv(operand1, operand2, *args, **kwargs):
        tile_size, ret_type, operand1, operand2 = ExtensionOverrides.binary_elementwise_common(operand1, operand2)
        shape = f"vector<{tile_size}x{ret_type}>" if tile_size > 1 else ret_type

        if ret_type.startswith("f"):
            raise ValueError("truncdiv is strictly for integers. Use truediv for floats.")
        
        # arith.divsi: Signed Integer Division (Result is truncated)
        return f'arith.divsi %{operand1}, %{operand2} : {shape}', [tile_size, ret_type]

    @staticmethod
    def floordiv(operand1, operand2, *args, **kwargs):
        tile_size, ret_type, operand1, operand2 = ExtensionOverrides.binary_elementwise_common(operand1, operand2)
        shape = f"vector<{tile_size}x{ret_type}>" if tile_size > 1 else ret_type

        if ret_type.startswith("f"):
             # Float의 floor division은 보통 divf 후 floor를 하므로 여기선 정수만 처리
             raise ValueError("floordiv implementation expects integers based on definition.")

        # arith.floordivsi: Floor Division for Signed Integers
        return f'arith.floordivsi %{operand1}, %{operand2} : {shape}', [tile_size, ret_type]

    @staticmethod
    def truediv(operand1, operand2, *args, **kwargs):
        tile_size, ret_type, operand1, operand2 = ExtensionOverrides.binary_elementwise_common(operand1, operand2)
        shape = f"vector<{tile_size}x{ret_type}>" if tile_size > 1 else ret_type

        if not ret_type.startswith("f"):
            raise ValueError(f"truediv expects float inputs, but got {ret_type}. Use int_truediv for integers.")

        return f'arith.divf %{operand1}, %{operand2} : {shape}', [tile_size, ret_type]

    @staticmethod
    def int_truediv(operand1, operand2, *args, **kwargs):
        """
        True division for Integers (Int -> Float).
        Promotes integers to floats, then performs floating-point division.
        """
        tile_size, src_type, operand1, operand2 = ExtensionOverrides.binary_elementwise_common(operand1, operand2)
        if not src_type.startswith("f"):
            target_float_type = "f32"
            operand1 = ops.to_dtype(operand1, target_float_type)
            operand2 = ops.to_dtype(operand2, target_float_type)
            src_type = target_float_type

        result = ops.truediv(operand1, operand2)
        return result, V.kernel.var_info[result]

    @staticmethod
    def mod(operand1, operand2, *args, **kwargs):
        tile_size, ret_type, operand1, operand2 = ExtensionOverrides.binary_elementwise_common(operand1, operand2)
        shape = f"vector<{tile_size}x{ret_type}>" if tile_size > 1 else ret_type
        if ret_type[0] == "f":
            raise NotImplementedError("Not support remainder operation for floating point")
        else:
            opcode = f'arith.remsi'
        return f'{opcode} %{operand1}, %{operand2} : {shape}', [tile_size, ret_type]

    @staticmethod
    def remainder(operand1, operand2, *args, **kwargs):
        tile_size, ret_type, operand1, operand2 = ExtensionOverrides.binary_elementwise_common(operand1, operand2)
        shape = f"vector<{tile_size}x{ret_type}>" if tile_size > 1 else ret_type

        if ret_type.startswith("f"):
            opcode = 'arith.remf'
        else:
            opcode = 'arith.remsi' # Signed Integer Remainder (LHS sign)

        return f'{opcode} %{operand1}, %{operand2} : {shape}', [tile_size, ret_type]

    @staticmethod
    def square(operand, *args, **kwargs):
        result = ops.mul(operand, operand)
        return result, V.kernel.var_info[result]

    @staticmethod
    def fma(operand1, operand2, operand3, *args, **kwargs):
        result = ops.mul(operand1, operand2)
        result = ops.add(result, operand3)
        return result, V.kernel.var_info[result]

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # PyTorchSim specific operations 

    @staticmethod
    def alloc(size, src_type, *args, **kwargs):
        return f"memref.alloc() : memref<{size}x{src_type}>", [size, src_type]

    @staticmethod
    def extractelement(operand, idx, *args, **kwargs):
        op_type = V.kernel.var_info[operand]
        tile_size = op_type[0]
        dtype = op_type[1]
        shape = f"vector<{tile_size}x{dtype}>" if tile_size > 1 else dtype
        return f"vector.extract %{operand}[{idx}]: {dtype} from {shape}", [1, dtype]

    @staticmethod
    def ext(operand, dtype, *args, **kwargs):
        op_type = V.kernel.var_info[operand]
        shape = f"vector<{op_type[0]}x{op_type[1]}>" if op_type[0] > 1 else f"{op_type[1]}"
        target_type = f"vector<{op_type[0]}x{dtype}>" if op_type[0] > 1 else f"{dtype}"
        if op_type[0] == "f":
            opcode = f'arith.extf'
        else:
            opcode = f'arith.extui'
        return f'{opcode} %{operand} : {shape} to {target_type}', [op_type[0], dtype]

    @staticmethod
    def to_bool(operand, *args, **kwargs):
        tile_size, ret_type = V.kernel.var_info[operand]
        if ret_type == "i1":
            return operand, [tile_size, ret_type]

        const_zero = ops.constant(0, ret_type)
        if tile_size > 1:
            const_zero = ops.broadcast(const_zero, tile_size)
        ret = ops.ne(operand, const_zero)
        return ret, [tile_size, "i1"]
    @staticmethod
    def step(size, dtype, *args, **kwargs):
        index_shape = f"vector<{size}x{dtype}>"
        return f"vector.step : {index_shape}", [size, dtype]

    @staticmethod
    def index_cast(operand, target_type, *args, **kwrags):
        op_type = V.kernel.var_info[operand]
        src_shape = f"vector<{op_type[0]}x{op_type[1]}>" if op_type[0] > 1 else op_type[1]
        des_shape = f"vector<{op_type[0]}x{target_type}>" if op_type[0] > 1 else target_type
        return f"arith.index_cast %{operand} : {src_shape} to {des_shape}", [op_type[0], target_type]

    @staticmethod
    def shape_cast(operand, src_shape, dst_shape, *args, **kwargs):
        operand_type = V.kernel.var_info[operand]
        return f"vector.shape_cast %{operand} : {src_shape} to {dst_shape}", operand_type

    @staticmethod
    def multi_reduction(acc, init, vec_size, red_size, red_shape, red_type, type_name, *args, **kwargs):
        if red_size == 1:
            final_reduced_shape = f"{type_name}"
            line = reduction_combine_vec(red_type, acc, init, axis=0, shape=red_shape, reduced_shape=final_reduced_shape)
        else:
            final_reduced_shape = f"vector<{red_size}x{type_name}>"
            new_vshape= f"vector<{vec_size//red_size}x{red_size}x{type_name}>"
            value = ops.shape_cast(acc, red_shape, new_vshape)
            line = reduction_combine_vec(red_type, value, init, axis=0, shape=new_vshape, reduced_shape=final_reduced_shape)
        return line, [red_size, type_name]

    @staticmethod
    def _load(compute_vec_size, mlir_dtype, buffer, indices, buffer_shape, *args, **kwargs):
        if compute_vec_size == 1:
            vshape = f"{mlir_dtype}"
            operation = "affine.load"
            line = f"{operation} %{buffer}[{indices}] : {buffer_shape}"
        else:
            vshape = f"vector<{compute_vec_size}x{mlir_dtype}>"
            operation = "affine.vector_load"
            line = f"{operation} %{buffer}[{indices}] : {buffer_shape}, {vshape}"
        return line, [compute_vec_size, mlir_dtype]

    @staticmethod
    def _store(operand, buffer, indices, buffer_shape, *args, buffer_name=None, **kwargs):
        compute_vec_size, mlir_dtype = V.kernel.var_info[operand][0], V.kernel.var_info[operand][1]

        if compute_vec_size == 1:
            vshape = f"{mlir_dtype}"
            operation = "affine.store"
            line = f"{operation} %{operand}, %{buffer}[{indices}] : {buffer_shape}"
        else:
            vshape = f"vector<{compute_vec_size}x{mlir_dtype}>"
            operation = "affine.vector_store"
            line = f"{operation} %{operand}, %{buffer}[{indices}] : {buffer_shape}, {vshape}"

        if buffer_name is not None:
            return common.DeferredLine(buffer_name, line), [None, None]
        else:
            return line, [None, None]