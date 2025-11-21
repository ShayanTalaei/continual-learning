import torch
from flashinfer import (
    BatchDecodeWithPagedKVCacheWrapper,
    BatchPrefillWithPagedKVCacheWrapper,
    cascade,
)
from torch import Tensor

from tokasaurus.model.types import (
    AttentionInfo,
    DeviceType,
    PageInformation,
    WrapperCollection,
)
from dataclasses import replace


def create_workspace_buffer(device: DeviceType):
    # flashinfer recommends a 128MB buffer
    return torch.empty(
        128 * 1024 * 1024,
        dtype=torch.uint8,
        device=device,
    )


def create_wrappers(
    device: DeviceType,
    num_attention_heads: int,
    num_key_value_heads: int,
    workspace_buffer: Tensor | None = None,
):
    if workspace_buffer is None:
        workspace_buffer = create_workspace_buffer(device)

    gqa_ratio = num_attention_heads // num_key_value_heads

    # NOTE: I think it's ok to reuse the buffers across both wrappers
    prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(workspace_buffer)
    hydragen_wrapper = BatchPrefillWithPagedKVCacheWrapper(workspace_buffer)
    decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer, use_tensor_cores=gqa_ratio >= 4
    )

    return WrapperCollection(
        prefill_wrapper=prefill_wrapper,
        hydragen_wrapper=hydragen_wrapper,
        decode_wrapper=decode_wrapper,
    )


def create_wrappers_for_cudagraph(
    device: DeviceType,
    num_attention_heads: int,
    num_key_value_heads: int,
    num_decode_sequences: int,
    max_kv_indices: int,
    workspace_buffer: Tensor | None = None,
):
    if workspace_buffer is None:
        workspace_buffer = create_workspace_buffer(device)

    gqa_ratio = num_attention_heads // num_key_value_heads

    decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer,
        use_tensor_cores=gqa_ratio >= 4,
        use_cuda_graph=True,
        paged_kv_indptr_buffer=torch.empty(
            num_decode_sequences + 1,
            dtype=torch.int32,
            device=device,
        ),
        paged_kv_indices_buffer=torch.empty(
            max_kv_indices, dtype=torch.int32, device=device
        ),
        paged_kv_last_page_len_buffer=torch.empty(
            num_decode_sequences, dtype=torch.int32, device=device
        ),
    )

    return WrapperCollection(
        prefill_wrapper=None,
        hydragen_wrapper=None,
        decode_wrapper=decode_wrapper,
    )


def append_to_kv_cache(
    token_indices: Tensor,
    key: Tensor,
    value: Tensor,
    k_cache: Tensor,
    v_cache: Tensor,
):
    """
    Important to back out of torch compile for this op, since the compiler
    seemed to be making a copy of the cache, taking a lot of mem/time.
    """

    _, num_key_value_heads, head_dim = key.shape

    flat_k_cache = k_cache.view(-1, num_key_value_heads, head_dim)
    flat_v_cache = v_cache.view(-1, num_key_value_heads, head_dim)

    flat_k_cache[token_indices] = key
    flat_v_cache[token_indices] = value


def _merge_attention_outputs_with_lse(
    out_norm: Tensor,
    lse_norm: Tensor,
    out_cart: Tensor,
    lse_cart: Tensor,
) -> tuple[Tensor, Tensor]:
    """
    Merge two attention outputs using LSE (log-sum-exp) for numerical stability.
    Returns both merged output and merged LSE.
    """
    # Ensure shapes match
    assert out_norm.shape == out_cart.shape, (
        f"Shape mismatch in merge: out_norm={out_norm.shape}, out_cart={out_cart.shape}"
            )
    
    # Cast LSE to output dtype
    lse_norm = lse_norm.to(out_norm.dtype)
    lse_cart = lse_cart.to(out_cart.dtype)
    
    # Ensure LSE shapes are compatible for broadcasting
    # LSE should be [num_tokens] or [num_tokens, num_heads] depending on FlashInfer version
    # Output should be [num_tokens, num_heads, head_dim] or [num_tokens, num_heads * head_dim]
    if lse_norm.shape != lse_cart.shape:
        # Try to broadcast - if one is [L] and other is [L, H], expand the first
        if len(lse_norm.shape) < len(lse_cart.shape):
            while lse_norm.shape != lse_cart.shape:
                lse_norm = lse_norm.unsqueeze(-1)
        elif len(lse_cart.shape) < len(lse_norm.shape):
            while lse_cart.shape != lse_norm.shape:
                lse_cart = lse_cart.unsqueeze(-1)
    
    # logZ = log(exp(lse_norm) + exp(lse_cart)) in a stable way
    logZ = torch.logaddexp(lse_norm, lse_cart)
    
    # Mixture weights for the two groups
    w_norm = torch.exp(lse_norm - logZ)
    w_cart = torch.exp(lse_cart - logZ)
    
    # Ensure weights have same number of dims as outputs for broadcasting
    while len(w_norm.shape) < len(out_norm.shape):
        w_norm = w_norm.unsqueeze(-1)
    while len(w_cart.shape) < len(out_cart.shape):
        w_cart = w_cart.unsqueeze(-1)
    
    merged_output = out_norm * w_norm + out_cart * w_cart
    
    return merged_output, logZ


def _run_wrapper(
    wrapper,
    q: Tensor,
    page_info: PageInformation,
    k_cache: Tensor,
    v_cache: Tensor,
    num_kv_heads: int,
    num_qo_heads: int,
    head_dim: int,
    page_size: int,
    causal: bool,
    is_decode: bool,
    return_lse: bool = False,
) -> tuple[Tensor, Tensor | None]:
    """Helper to run a single route wrapper."""
    
    # If no blocks in this route, return zeros
    # Note: This happens if a route is completely empty for all sequences
    if page_info.kv_indices.numel() == 0:
        # Output shape depends on wrapper type
        # Prefill: [num_tokens, num_heads, head_dim]
        # Decode: [batch_size, num_heads, head_dim]
        num_tokens = q.shape[0] # This is correct for both prefill (all tokens) and decode (batch size)
        
        out = torch.zeros(
            (num_tokens, num_qo_heads, head_dim), 
            device=q.device, 
            dtype=q.dtype
        )
        # LSE shape: [num_tokens, num_heads]
        lse = torch.full(
            (num_tokens, num_qo_heads), 
            float("-inf"), 
            device=q.device, 
            dtype=torch.float32
        ) if return_lse else None
        return out, lse

    # Plan
    if is_decode:
        wrapper.plan(
            indptr=page_info.kv_indptr,
            indices=page_info.kv_indices,
            last_page_len=page_info.kv_last_page_len,
            num_kv_heads=num_kv_heads,
            num_qo_heads=num_qo_heads,
            head_dim=head_dim,
            page_size=page_size,
            q_data_type=q.dtype,
            kv_data_type=k_cache.dtype,
            non_blocking=False,
        )
    else:
        # For prefill/hydragen, we MUST use the ORIGINAL qo_indptr to ensure
        # correct query-to-sequence mapping, even if kv_indptr is split.
        wrapper.plan(
            qo_indptr=page_info.qo_indptr, 
            paged_kv_indptr=page_info.kv_indptr,
            paged_kv_indices=page_info.kv_indices,
            paged_kv_last_page_len=page_info.kv_last_page_len,
            num_kv_heads=num_kv_heads,
            num_qo_heads=num_qo_heads,
            head_dim_qk=head_dim,
            page_size=page_size,
            q_data_type=q.dtype,
            kv_data_type=k_cache.dtype,
            causal=causal,
            non_blocking=False,
        )

    # Run
    if return_lse:
        if hasattr(wrapper, "run_return_lse"):
            return wrapper.run_return_lse(q=q, paged_kv_cache=(k_cache, v_cache))
        else:
             # Fallback (potentially inaccurate merging)
            out = wrapper.run(q=q, paged_kv_cache=(k_cache, v_cache))
            return out, torch.zeros(out.shape[0], device=q.device, dtype=torch.float32)
    else:
        out = wrapper.run(q=q, paged_kv_cache=(k_cache, v_cache))
        return out, None


def _standard_attention(
    ragged_q: Tensor,
    ragged_k: Tensor,
    ragged_v: Tensor,
    k_cache: Tensor,
    v_cache: Tensor,
    attn_info: AttentionInfo,
    wrappers: WrapperCollection,
) -> Tensor:
    """
    Original single-route attention logic (backward compatible).
    Uses the already-planned wrappers from the model's plan() method.
    """
    prefill_q, hydragen_q, decode_q = attn_info.split_q(ragged_q)

    # the key difference between the hydragen shared
    # prefix attention and normal prefill
    # is that hydragen does not have a causal mask
    if prefill_q.numel() > 0:
        prefill_wrapper = wrappers.prefill_wrapper
        assert prefill_wrapper is not None
        true_prefill_output = prefill_wrapper.run(
            q=prefill_q, paged_kv_cache=(k_cache, v_cache)
        )
    else:
        true_prefill_output = prefill_q

    # decode
    if decode_q.numel() > 0:
        decode_wrapper = wrappers.decode_wrapper
        assert decode_wrapper is not None
        decode_output, decode_lse = decode_wrapper.run_return_lse(
            q=decode_q, paged_kv_cache=(k_cache, v_cache)
        )
    else:
        decode_output = decode_q
        decode_lse = None

    if hydragen_q.numel() > 0:
        hydragen_wrapper = wrappers.hydragen_wrapper
        assert hydragen_wrapper is not None
        shared_prefill_output, shared_prefill_lse = hydragen_wrapper.run_return_lse(
            q=hydragen_q, paged_kv_cache=(k_cache, v_cache)
        )

        # Unique (decode)
        assert attn_info.hydragen_info is not None
        n_mixed = attn_info.hydragen_info.num_tokens
        assert decode_lse is not None
        unique_lse = decode_lse[:n_mixed]
        unique_out = decode_output[:n_mixed]

        aggregate, _ = cascade.merge_state(
            shared_prefill_output, shared_prefill_lse, unique_out, unique_lse
        )

        true_decode_out = decode_output[n_mixed:]
        output = torch.cat([true_prefill_output, aggregate, true_decode_out], dim=0)

    else:
        output = torch.cat([true_prefill_output, decode_output], dim=0)

    return output


def _multi_route_attention(
    ragged_q: Tensor,  # rotated queries
    ragged_q_unrot: Tensor,  # unrotated queries
    ragged_k: Tensor,
    ragged_v: Tensor,
    k_cache: Tensor,
    v_cache: Tensor,
    attn_info: AttentionInfo,
    wrappers: WrapperCollection,
) -> Tensor:
    """
    Multi-route attention with unrotated queries for cartridges.
    """
    cartridge_block_indices = attn_info.cartridge_block_indices
    assert cartridge_block_indices is not None, "cartridge_block_indices must be set"
    page_size = attn_info.page_size
    
    # Split queries
    prefill_q, hydragen_q, decode_q = attn_info.split_q(ragged_q)
    prefill_q_unrot, hydragen_q_unrot, decode_q_unrot = attn_info.split_q(ragged_q_unrot)
    
    # Common dimensions
    num_kv_heads = ragged_k.shape[1]
    
    outputs = []
    
    # --- 1. Prefill phase ---
    if prefill_q.numel() > 0:
        prefill_norm, prefill_cart = attn_info.prefill_info.split_by_cartridge_blocks(
            cartridge_block_indices, page_size
        )
        
        num_qo_heads = prefill_q.shape[1]
        head_dim = prefill_q.shape[-1]
        wrapper = wrappers.prefill_wrapper
        assert wrapper is not None

        # Run Normal Route
        out_norm, lse_norm = _run_wrapper(
            wrapper, prefill_q, prefill_norm, k_cache, v_cache,
            num_kv_heads, num_qo_heads, head_dim, page_size, 
            causal=True, is_decode=False, return_lse=True
        )
        
        # Run Cartridge Route
        out_cart, lse_cart = _run_wrapper(
            wrapper, prefill_q_unrot, prefill_cart, k_cache, v_cache,
            num_kv_heads, num_qo_heads, head_dim, page_size, 
            causal=True, is_decode=False, return_lse=True
        )
        
        # Merge
        assert lse_norm is not None and lse_cart is not None
        merged_out, _ = _merge_attention_outputs_with_lse(out_norm, lse_norm, out_cart, lse_cart)
        outputs.append(merged_out)
    else:
        outputs.append(prefill_q) # Append empty tensor if needed, or nothing
    
    # --- 2. Decode phase ---
    decode_output = None
    decode_lse = None
    
    if decode_q.numel() > 0:
        decode_norm, decode_cart = attn_info.decode_info.split_by_cartridge_blocks(
            cartridge_block_indices, page_size
        )
        
        num_qo_heads = decode_q.shape[1]
        head_dim = decode_q.shape[-1]
        wrapper = wrappers.decode_wrapper
        assert wrapper is not None
        
        # Run Normal Route
        out_norm, lse_norm = _run_wrapper(
            wrapper, decode_q, decode_norm, k_cache, v_cache,
            num_kv_heads, num_qo_heads, head_dim, page_size, 
            causal=True, is_decode=True, return_lse=True
        )
        
        # Run Cartridge Route
        out_cart, lse_cart = _run_wrapper(
            wrapper, decode_q_unrot, decode_cart, k_cache, v_cache,
            num_kv_heads, num_qo_heads, head_dim, page_size, 
            causal=True, is_decode=True, return_lse=True
        )
        
        # Merge
        assert lse_norm is not None and lse_cart is not None
        decode_output, decode_lse = _merge_attention_outputs_with_lse(out_norm, lse_norm, out_cart, lse_cart)
        outputs.append(decode_output)
    else:
        outputs.append(decode_q)
    
    # --- 3. Hydragen phase ---
    if hydragen_q.numel() > 0:
        assert attn_info.hydragen_info is not None
        hydragen_norm, hydragen_cart = attn_info.hydragen_info.split_by_cartridge_blocks(
            cartridge_block_indices, page_size
        )
        
        num_qo_heads = hydragen_q.shape[1]
        head_dim = hydragen_q.shape[-1]
        wrapper = wrappers.hydragen_wrapper
        assert wrapper is not None
        
        # Run Normal Route (Shared Prefix)
        sp_out_norm, sp_lse_norm = _run_wrapper(
            wrapper, hydragen_q, hydragen_norm, k_cache, v_cache,
            num_kv_heads, num_qo_heads, head_dim, page_size, 
            causal=False, is_decode=False, return_lse=True
        )
        
        # Run Cartridge Route (Shared Prefix)
        sp_out_cart, sp_lse_cart = _run_wrapper(
            wrapper, hydragen_q_unrot, hydragen_cart, k_cache, v_cache,
            num_kv_heads, num_qo_heads, head_dim, page_size, 
            causal=False, is_decode=False, return_lse=True
        )
        
        # Merge Shared Prefix
        assert sp_lse_norm is not None and sp_lse_cart is not None
        shared_prefill_output, shared_prefill_lse = _merge_attention_outputs_with_lse(
            sp_out_norm,
            sp_lse_norm,
            sp_out_cart,
            sp_lse_cart,
        )
        
        # Get Unique (Decode) Part
        assert attn_info.hydragen_info is not None
        n_mixed = attn_info.hydragen_info.num_tokens
        
        # We reuse the decode output computed above
        if decode_output is not None:
            assert decode_lse is not None
            unique_lse = decode_lse[:n_mixed]
            unique_out = decode_output[:n_mixed]
        else:
            # Fallback (should not happen)
            unique_out = decode_q_unrot[:n_mixed]
            unique_lse = torch.full((n_mixed, num_qo_heads), float("-inf"), device=decode_q.device)
        
        # Cascade Merge
        aggregate, _ = cascade.merge_state(
            shared_prefill_output, shared_prefill_lse, unique_out, unique_lse
        )
        
        # Assemble Final Output
        # Current outputs list: [prefill, decode]
        # We need to replace decode part with: [aggregate, remaining_decode]
        
        # Split the decode output we appended earlier
        true_decode_out = decode_output[n_mixed:] if decode_output is not None else torch.empty(0, device=decode_q.device, dtype=decode_q.dtype)
        
        # Replace the last element (decode)
        outputs.pop()
        outputs.append(aggregate)
        if true_decode_out.numel() > 0:
            outputs.append(true_decode_out)
    
    return torch.cat(outputs, dim=0)


def tokasaurus_attention(
    ragged_q: Tensor,
    ragged_q_unrot: Tensor | None,  # NEW: unrotated queries
    ragged_k: Tensor,
    ragged_v: Tensor,
    k_cache: Tensor,
    v_cache: Tensor,
    attn_info: AttentionInfo,
    wrappers: WrapperCollection,
    use_unrotated_queries: bool = False,  # config flag
) -> Tensor:
    """
    Assumes rope has been already applied to ragged_q.
    ragged_q_unrot should be unrotated (before RoPE).
    """
    append_to_kv_cache(
        token_indices=attn_info.append_kv_token_indices,
        key=ragged_k,
        value=ragged_v,
        k_cache=k_cache,
        v_cache=v_cache,
    )
    
    # Check if multi-route attention is needed
    has_cartridges = attn_info.has_cartridges()
    use_multi_route = use_unrotated_queries and has_cartridges and ragged_q_unrot is not None
    
    if not use_multi_route:
        # Standard single-route attention (backward compatible)
        return _standard_attention(
            ragged_q, ragged_k, ragged_v, k_cache, v_cache, attn_info, wrappers
        )
    
    # Multi-route attention with unrotated queries
    assert ragged_q_unrot is not None, "ragged_q_unrot must be provided for multi-route attention"
    return _multi_route_attention(
        ragged_q, ragged_q_unrot, ragged_k, ragged_v, k_cache, v_cache, attn_info, wrappers
    )
