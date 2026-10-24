"""Memory controls and actual text/image attention maps for frozen Iris guidance."""

from contextlib import contextmanager, nullcontext
import math

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


@contextmanager
def gradient_memory(model, offload=True):
    """Enable upstream block checkpointing without changing child eval modes."""
    training = model.training
    policy = getattr(model, "activation_checkpointing", None)
    try:
        if policy is not None:
            model.activation_checkpointing = "full"
            model.training = True  # only the root's checkpoint-policy gate
        context = (torch.autograd.graph.save_on_cpu(pin_memory=True)
                   if offload and next(model.parameters()).device.type == "cuda" else nullcontext())
        with context:
            yield
    finally:
        model.training = training
        if policy is not None:
            model.activation_checkpointing = policy


def subject_token_indices(tokenizer, full_prompts, subjects):
    """Match character spans in caption tokens (Iris slices off the chat prefix)."""
    result = []
    for prompt, subject in zip(full_prompts, subjects):
        start = prompt.casefold().rfind(subject.casefold())
        if not subject.strip() or start < 0:
            raise ValueError(f"Subject words {subject!r} must occur in the corresponding prompt")
        encoded = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
        positions = [i for i, (left, right) in enumerate(encoded["offset_mapping"])
                     if right > start and left < start + len(subject)]
        if not positions or max(positions) >= 297:
            raise ValueError("Subject words must fit inside Iris's caption token budget")
        result.append(positions)
    return result


class _CapturedAttention(Exception):
    pass


def _single_stream_map(block, args, positions, resolution):
    """Joint attention probability on subject tokens, averaged over all heads.

    Pool queries to <=32x32 before evaluating their full image+text key
    normalization. Checkpoint chunks so the quadratic attention graph is not
    retained. The normal denoiser forward is not replaced by this calculation.
    """
    from iris3b.nn.modulation import modulate
    from iris3b.nn.rope import apply_rope

    x, y, cond, rope_img, rope_txt = args
    batch, count, dim = x.shape
    text_count = y.shape[1]
    s1, c1, *_ = block.adaln(cond).chunk(6, dim=-1)
    h = modulate(block.norm1(torch.cat([y, x], dim=1)), s1, c1)
    if block.qkv is not None:
        q, k, _ = block.qkv(h).reshape(batch, text_count + count, 3, block.num_heads, block.head_dim).unbind(2)
    else:
        q = block.q_proj(h).reshape(batch, text_count + count, block.num_heads, block.head_dim)
        k = block.k_proj(h).reshape(batch, text_count + count, block.num_kv_heads, block.head_dim)
    q, k = block.q_norm(q), block.k_norm(k)
    qx, kx = apply_rope(q[:, text_count:], k[:, text_count:], rope_img)
    ky = k[:, :text_count]
    if rope_txt is not None:
        _, ky = apply_rope(q[:, :text_count], ky, rope_txt)
    keys = torch.cat([ky, kx], dim=1).transpose(1, 2)
    if block.num_kv_heads != block.num_heads:
        keys = keys.repeat_interleave(block.num_heads // block.num_kv_heads, dim=1)
    grid = math.isqrt(count)
    if grid * grid != count:
        raise ValueError("Attention guidance currently requires square images")
    side = min(resolution, grid)
    queries = qx.permute(0, 2, 3, 1).reshape(batch * block.num_heads, block.head_dim, grid, grid)
    queries = F.adaptive_avg_pool2d(queries.float(), (side, side))
    queries = queries.reshape(batch, block.num_heads, block.head_dim, side * side).transpose(-2, -1)
    selected = torch.tensor(positions, device=x.device)
    if max(positions) >= text_count:
        raise ValueError("Subject token is outside the model text sequence")

    def probability(query, key):
        with torch.autocast(x.device.type, enabled=False):
            logits = query.float() @ key.float().transpose(-2, -1) / math.sqrt(block.head_dim)
            return logits.softmax(-1).index_select(-1, selected).sum(-1).mean(1)

    pieces = []
    for chunk in queries.split(64, dim=-2):
        if torch.is_grad_enabled():
            pieces.append(checkpoint(probability, chunk, keys, use_reentrant=False))
        else:
            pieces.append(probability(chunk, keys))
    return torch.cat(pieces, dim=-1).reshape(batch, 1, side, side)


def attention_map(model, image, time, embeddings, mask, positions, *, resolution=32):
    """Capture the first joint single-stream block and stop before pixel refinement."""
    blocks = [block for block in model.blocks if hasattr(block, "q_norm") and hasattr(block, "adaln")]
    if not blocks:
        raise ValueError("Anagram-MTL requires Iris single-stream joint-attention blocks")
    captured = []

    def capture(block, args):
        # Compute the map after leaving the aborted checkpoint frame. A map
        # built in this hook would be inside that frame, whose backward
        # recomputation would then run the block after the hook is removed.
        captured.append(args)
        raise _CapturedAttention()

    hook = blocks[0].register_forward_pre_hook(capture)
    try:
        try:
            model(image, time, embeddings, y_mask=mask)
        except _CapturedAttention:
            pass
    finally:
        hook.remove()
    if not captured:
        raise RuntimeError("The configured attention capture block was not reached")
    return _single_stream_map(blocks[0], captured[0], positions, resolution)


def align_attention(value, view, size):
    """Align spatial probabilities using the saved full-resolution view mapping."""
    full = F.interpolate(value.float(), (size, size), mode="bilinear", align_corners=False)[0]
    mapped = view.inverse_view(full.expand(3, -1, -1).clone()).abs().mean(0)[None, None]
    return F.adaptive_avg_pool2d(mapped, value.shape[-2:])


def overlap_loss(first, second, target=0.45):
    denominator = (first + second).sum().clamp_min(1e-12)
    return (target - torch.minimum(first, second).sum() / denominator).abs()
