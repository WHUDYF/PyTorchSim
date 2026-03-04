from typing import List, Optional
import math
import itertools

import sympy
from torch._inductor.ir import IRNode

from PyTorchSimFrontend.mlir import mlir_common
from PyTorchSimFrontend.mlir.mlir_template import MLIRTemplate, MLIRTemplateKernel


TEMPLATE = r"""
{{kernel.def_global_vars()}}
func.func @{{ KERNEL_NAME }} {{kernel.def_kernel(inputs=INPUT_NAMES, outputs=[Y], names_str=NAMES_STR, input_reorder=input_reorder)}} {
{%- for buffer_name, tile_desc in UNIQUE_BUFFER_TILE_DESCS.items() %}
  {{ kernel.def_sram_buffer(buffer_name, tile_desc, indent_size=2) }}
{%- endfor %}
  {{ kernel.def_local_vars(indent_size=2) }}

  affine.for %cat_block = 0 to 1 step 1 {
{%- for d in range(RANK-1) %}
    affine.for %index{{ OUTPUT_DIM[d] }} = 0 to {{ OUTPUT_SIZES[d] }} step {{ TILE_SIZES[d] }} {
{%- endfor %}
{%- for i in range(NUM_INPUTS) %}
      // Input tensor{{ i }}
      affine.for %index_local{{ DIM }}_{{ i }} = 0 to {{ INPUT_SIZES[i][DIM] }} step {{ INPUT_TILE_SIZES_DIM[i] }} {
        %index{{ DIM }}_{{i}} = affine.apply affine_map<(d0) -> (d0 + {{ CUMULATIVE_OFFSETS[i] }})> (%index_local{{ DIM }}_{{ i }})
        {{ kernel.def_dma_op("MVIN", INPUT_BUFFER_NAMES[i], INPUT_IDXS[i], INPUT_TILE_DESCS[i], indent_size=INDENT_SIZE) }}
        {{ kernel.def_dma_op("MVOUT", OUT_DVAR, OUTPUT_IDXS[i], OUTPUT_TILE_DESCS[i], indent_size=INDENT_SIZE) }}
      } { inner_loop=true }
{%- endfor %}

{%- for d in range(RANK-1) %}
    } { outer_loop=true }
{%- endfor %}
  } { outer_loop=true }
  return
}
"""


class MLIRCatTemplate(MLIRTemplate):
    def __init__(self, input_nodes, layout, dim):
        super().__init__("kernel", input_nodes, layout)
        self.dim = dim

    def render(
        self,
        kernel: MLIRTemplateKernel,
        template_buffer_node=None,
        epilogue_nodes: Optional[List[IRNode]] = None,
        tile_info=None,
        **kwargs,
    ):
        # Extract info
        input_nodes = self.input_nodes
        y = self.output_node
        num_inputs = len(self.input_nodes)
        rank = len(y.get_size())

        input_sizes = [x.get_size() for x in input_nodes]
        output_sizes = [sz for dim, sz in enumerate(y.get_size()) if dim != self.dim]
        output_dim = [dim for dim, sz in enumerate(y.get_size()) if dim != self.dim]
        tile_sizes = tile_info if tile_info is not None else [1] * len(output_sizes)
        output_strides = y.get_layout().stride

        # Calculate input tile sizes
        input_tile_sizes_dim = self._calculate_input_tile_sizes(
            kernel, input_sizes, tile_sizes, num_inputs, rank
        )
        buffer_name_to_template_name, input_buffer_names = self._build_buffer_mapping(input_nodes)
        input_tile_descs, output_tile_descs, unique_tile_descs = self._build_tile_descriptors(
            kernel, input_nodes, input_sizes, input_tile_sizes_dim, tile_sizes, rank, input_buffer_names, y
        )

        input_idxs, output_idxs, cumulative_offsets = self._build_index_expressions(
            input_nodes, input_sizes, output_strides, rank, num_inputs
        )

        # Map unique buffer names to their tile descriptors for template
        unique_buffer_tile_descs = {}
        for actual_name, template_name in buffer_name_to_template_name.items():
            if actual_name in unique_tile_descs:
                unique_buffer_tile_descs[template_name] = unique_tile_descs[actual_name]

        names_str = ", ".join(input_buffer_names + ["Y"])
        indent_size = 2 + (rank - 1) * 2 + 4

        kernel.render_options = dict(
            KERNEL_NAME=self.name,
            kernel=kernel,
            Y=y,
            OUT_DVAR="Y",
            NAMES_STR=names_str,
            INPUT_NAMES=input_nodes,
            INPUT_BUFFER_NAMES=input_buffer_names,
            NUM_INPUTS=num_inputs,
            RANK=rank,
            DIM=self.dim,
            INPUT_SIZES=input_sizes,
            OUTPUT_SIZES=output_sizes,
            OUTPUT_DIM=output_dim,
            TILE_SIZES=tile_sizes,
            INPUT_TILE_SIZES_DIM=input_tile_sizes_dim,
            INPUT_TILE_DESCS=input_tile_descs,
            OUTPUT_TILE_DESCS=output_tile_descs,
            UNIQUE_BUFFER_TILE_DESCS=unique_buffer_tile_descs,
            INPUT_IDXS=input_idxs,
            OUTPUT_IDXS=output_idxs,
            CUMULATIVE_OFFSETS=cumulative_offsets,
            INDENT_SIZE=indent_size,
            input_reorder=self.input_reorder,
        )

        code = self._template_from_string(TEMPLATE).render(**kernel.render_options)
        return code

    def get_tile_candidates(
        self,
        kernel: MLIRTemplateKernel,
        template_buffer_node=None,
        epilogue_nodes: Optional[List[IRNode]] = None,
        **kwargs,
    ):
        """Generate tile candidates for cat operation. Concat dimension always has tile size 1."""
        if template_buffer_node is not None:
            self.output_node = template_buffer_node

        y = self.output_node
        num_inputs = len(self.input_nodes)
        output_sizes = [sz for dim, sz in enumerate(y.get_size()) if dim != self.dim]
        num_non_dim_dims = len(output_sizes)

        if num_non_dim_dims == 0:
            return [[1]]

        tile_candidates = []
        dim_tile_candidates = []

        for dim_size in output_sizes:
            dim_candidates = []
            max_tile = min(dim_size, kernel.spad_info["spad_size"] // (kernel.vector_lane * kernel.precision * 2 * num_inputs))

            for mult in range(1, max_tile // kernel.vector_lane + 1):
                tile = mult * kernel.vector_lane
                if tile <= dim_size:
                    dim_candidates.append(tile)

            if max_tile > 0:
                for exp in range(int(math.log2(max_tile)) + 1):
                    tile = 2 ** exp
                    if tile <= dim_size and tile not in dim_candidates:
                        dim_candidates.append(tile)

            if dim_size not in dim_candidates:
                dim_candidates.append(dim_size)

            dim_tile_candidates.append(sorted(set(dim_candidates))[:5])

        for tile_combo in itertools.product(*dim_tile_candidates):
            total_elements = math.prod(tile_combo)
            total_spad_needed = total_elements * (num_inputs + 1) * kernel.precision

            if total_spad_needed <= kernel.spad_info["spad_size"] * kernel.vector_lane:
                tile_candidates.append(list(tile_combo))

        if not tile_candidates:
            tile_candidates = [[1] * num_non_dim_dims]

        tile_candidates.sort(key=lambda x: -math.prod(x))
        return tile_candidates[:4]

    def _calculate_input_tile_sizes(
        self, kernel, input_sizes, tile_sizes, num_inputs, rank
    ):
        """Calculate tile sizes for concat dimension for each input."""
        non_dim_tile_elements = math.prod(tile_sizes) if tile_sizes else 1
        non_dim_tile_spad = non_dim_tile_elements * kernel.precision
        max_spad_per_input = kernel.spad_info["spad_size"] * kernel.vector_lane // 2
        extra_concat_input = math.ceil(max_spad_per_input / non_dim_tile_spad) - num_inputs

        input_tile_sizes_dim = []
        for i in range(num_inputs):
            input_dim_size = input_sizes[i][self.dim]
            if extra_concat_input > 0 and non_dim_tile_elements > 0:
                max_tile_dim = min(input_dim_size, extra_concat_input)
                extra_concat_input -= max_tile_dim
            else:
                max_tile_dim = 1
            input_tile_sizes_dim.append(max_tile_dim)
        return input_tile_sizes_dim

    def _build_buffer_mapping(self, input_nodes):
        """Map actual buffer names to template buffer names """
        buffer_name_to_template_name = {}
        input_buffer_names = []
        for x in input_nodes:
            actual_name = x.get_name()
            template_name = buffer_name_to_template_name.setdefault(
                actual_name, f"X{len(buffer_name_to_template_name)}"
            )
            input_buffer_names.append(template_name)
        return buffer_name_to_template_name, input_buffer_names

    def _build_tile_descriptors(
        self, kernel, input_nodes, input_sizes, input_tile_sizes_dim, tile_sizes, rank, input_buffer_names, output_node
    ):
        """Build tile descriptors for each input and output."""
        input_tile_descs = []
        output_tile_descs = []
        unique_tile_descs = {}
        output_offset = output_node.get_layout().offset

        for i, x in enumerate(input_nodes):
            x_offset = x.get_layout().offset
            full_tile_sizes = []
            tile_size_idx = 0
            for d in range(rank):
                if d != self.dim:
                    full_tile_sizes.append(tile_sizes[tile_size_idx])
                    tile_size_idx += 1
                else:
                    full_tile_sizes.append(input_tile_sizes_dim[i])

            # Input tile descriptor
            input_tile_desc = mlir_common.MLIRMultiDimTile(
                full_tile_sizes,
                kernel.vector_lane,
                vlane_split_axis=rank - 1,
                vlane_stride=1
            )
            input_tile_desc.set_tile_size(full_tile_sizes)
            template_buffer_name = input_buffer_names[i]
            input_tile_desc.set_name(f"{template_buffer_name.lower()}_cat_tile")
            input_tile_desc.offset = x_offset
            input_tile_descs.append(input_tile_desc)

            # Output tile descriptor (same as input but with output offset)
            output_tile_desc = mlir_common.MLIRMultiDimTile(
                full_tile_sizes,
                kernel.vector_lane,
                vlane_split_axis=rank - 1,
                vlane_stride=1
            )
            output_tile_desc.set_tile_size(full_tile_sizes)
            output_tile_desc.set_name(f"{template_buffer_name.lower()}_cat_tile")
            output_tile_desc.offset = output_offset
            output_tile_descs.append(output_tile_desc)

            # Store unique tile desc by actual buffer name
            actual_name = x.get_name()
            if actual_name not in unique_tile_descs:
                unique_tile_descs[actual_name] = input_tile_desc

        return input_tile_descs, output_tile_descs, unique_tile_descs

    def _build_index_expressions(
        self, input_nodes, input_sizes, output_strides, rank, num_inputs
    ):
        """Build index expressions for input and output."""
        input_idxs = []
        output_idxs = []
        cumulative_offsets = [0]
        for i in range(num_inputs - 1):
            cumulative_offsets.append(cumulative_offsets[-1] + input_sizes[i][self.dim])

        for i, x in enumerate(input_nodes):
            x_stride = x.get_layout().stride
            x_offset = x.get_layout().offset
            if hasattr(x, 'data') and hasattr(x.data, 'dims'):
                # In case of PermuteView, the stride is permuted
                perm_dims = x.data.dims
                x_stride = [x_stride[perm_dims[d]] for d in range(rank)]

            input_idx = []
            output_idx = []
            for d in range(rank):
                if d != self.dim:
                    input_idx_symbol = sympy.Symbol(f"index{d}")
                    output_idx_symbol = sympy.Symbol(f"index{d}")
                else:
                    input_idx_symbol = sympy.Symbol(f"index_local{self.dim}_{i}")
                    output_idx_symbol = sympy.Symbol(f"index{self.dim}_{i}")
                input_idx.append(input_idx_symbol * x_stride[d])
                output_idx.append(output_idx_symbol * output_strides[d])
            input_idxs.append(input_idx)
            output_idxs.append(output_idx)

        return input_idxs, output_idxs, cumulative_offsets
