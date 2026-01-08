import math
import torch
import torch.nn.functional as F
from torch._inductor.decomposition import register_decomposition

aten = torch.ops.aten

@register_decomposition(aten._native_multi_head_attention.default)
def decompose_native_multi_head_attention(
    query,
    key,
    value,
    embed_dim: int,
    num_heads: int,
    qkv_weight,
    qkv_bias,
    proj_weight,
    proj_bias,
    mask=None,
    need_weights: bool = False,
):
    """
    Decompose _native_multi_head_attention into scaled_dot_product_attention operations.

    Based on F.scaled_dot_product_attention and nn.MultiheadAttention implementation:
    1. QKV projection (if needed - but query/key/value may already be projected)
    2. Reshape to multi-head format
    3. Scaled dot product: Q @ K^T / sqrt(head_dim)
    4. Softmax
    5. Attention @ V
    6. Reshape back and output projection
    """
    head_dim = embed_dim // num_heads
    scale_factor = 1.0 / math.sqrt(head_dim)

    # Get input shapes - assuming [batch, seq_len, embed_dim] format
    query_shape = query.shape
    if len(query_shape) == 3:
        # [batch, seq_len, embed_dim] format
        batch_size = query_shape[0]
        seq_len = query_shape[1]
    elif len(query_shape) == 2:
        # [seq_len, embed_dim] -> add batch dimension
        batch_size = 1
        seq_len = query_shape[0]
        query = query.unsqueeze(0)  # [1, seq_len, embed_dim]
        key = key.unsqueeze(0)
        value = value.unsqueeze(0)
    else:
        # Fallback: assume first dim is batch, second is seq_len
        batch_size = query_shape[0] if len(query_shape) > 0 else 1
        seq_len = query_shape[1] if len(query_shape) > 1 else query_shape[0]

    # Step 1: QKV projection (if query/key/value are not already projected)
    # In many cases, query/key/value are already projected, so we check if qkv_weight is used
    # For now, assume they might need projection
    # Note: In practice, _native_multi_head_attention often receives already projected inputs

    # Reshape for projection: [batch, seq_len, embed_dim] -> [batch*seq_len, embed_dim]
    if len(query.shape) == 3:
        query_flat = query.view(-1, embed_dim)
        key_flat = key.view(-1, embed_dim)
        value_flat = value.view(-1, embed_dim)
    else:
        query_flat = query
        key_flat = key
        value_flat = value

    # QKV projection using qkv_weight and qkv_bias
    # qkv_weight shape: [3*embed_dim, embed_dim] -> split into 3 parts
    # Split qkv_weight into Q, K, V weights
    qkv_weight_q, qkv_weight_k, qkv_weight_v = torch.split(qkv_weight, embed_dim, dim=0)
    if qkv_bias is not None:
        # qkv_bias shape: [3*embed_dim] -> split into 3 parts
        qkv_bias_q, qkv_bias_k, qkv_bias_v = torch.split(qkv_bias, embed_dim, dim=0)
    else:
        qkv_bias_q = qkv_bias_k = qkv_bias_v = None

    # Project Q, K, V
    q = torch.nn.functional.linear(query_flat, qkv_weight_q, qkv_bias_q)
    k = torch.nn.functional.linear(key_flat, qkv_weight_k, qkv_bias_k)
    v = torch.nn.functional.linear(value_flat, qkv_weight_v, qkv_bias_v)

    # Reshape back: [batch*seq_len, embed_dim] -> [batch, seq_len, embed_dim]
    q = q.view(batch_size, seq_len, embed_dim)
    k = k.view(batch_size, seq_len, embed_dim)
    v = v.view(batch_size, seq_len, embed_dim)

    # Step 2: Reshape to multi-head format
    # [batch, seq_len, embed_dim] -> [batch, seq_len, num_heads, head_dim]
    q = q.view(batch_size, seq_len, num_heads, head_dim)
    k = k.view(batch_size, seq_len, num_heads, head_dim)
    v = v.view(batch_size, seq_len, num_heads, head_dim)

    # Transpose to [batch, num_heads, seq_len, head_dim] for bmm
    # [batch, seq_len, embed_dim] -> [batch, seq_len, num_heads, head_dim]
    q = q.view(batch_size, seq_len, num_heads, head_dim)
    k = k.view(batch_size, seq_len, num_heads, head_dim)
    v = v.view(batch_size, seq_len, num_heads, head_dim)

    # Transpose to [batch, num_heads, seq_len, head_dim] for bmm
    q = q.transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]
    k = k.transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]
    v = v.transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]

    # Step 3: Scaled dot product attention
    # Scale Q
    q_scaled = q * scale_factor

    # Q @ K^T: [batch, num_heads, seq_len, head_dim] @ [batch, num_heads, head_dim, seq_len]
    # -> [batch, num_heads, seq_len, seq_len]
    k_transposed = k.transpose(-2, -1)  # [batch, num_heads, head_dim, seq_len]
    scores = torch.matmul(q_scaled, k_transposed)  # [batch, num_heads, seq_len, seq_len]

    # Step 4: Apply mask if provided
    if mask is not None:
        scores = scores + mask

    # Step 5: Softmax along the last dimension (seq_len dimension)
    # Stable softmax: subtract max, exp, divide by sum
    scores_max = scores.amax(dim=-1, keepdim=True)  # [batch, num_heads, seq_len, 1]
    scores_shifted = scores - scores_max
    scores_exp = scores_shifted.exp()
    scores_sum = scores_exp.sum(dim=-1, keepdim=True)  # [batch, num_heads, seq_len, 1]
    attn_weights = scores_exp / scores_sum  # [batch, num_heads, seq_len, seq_len]

    # Step 6: Attention @ V
    # [batch, num_heads, seq_len, seq_len] @ [batch, num_heads, seq_len, head_dim]
    # -> [batch, num_heads, seq_len, head_dim]
    attn_output = torch.matmul(attn_weights, v)

    # Step 7: Reshape back to [batch, seq_len, embed_dim]
    attn_output = attn_output.transpose(1, 2)  # [batch, seq_len, num_heads, head_dim]
    attn_output = attn_output.contiguous().view(batch_size, seq_len, embed_dim)

    # Step 8: Output projection
    attn_output_flat = attn_output.view(-1, embed_dim)
    output = torch.nn.functional.linear(attn_output_flat, proj_weight, proj_bias)
    output = output.view(batch_size, seq_len, embed_dim)

    if need_weights:
        # Return attention weights: [batch, num_heads, seq_len, seq_len] -> [batch, seq_len, seq_len]
        attn_weights_mean = attn_weights.mean(dim=1)  # Average over heads
        return output, attn_weights_mean
    else:
        return (output, None)