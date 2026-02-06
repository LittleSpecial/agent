from __future__ import annotations

from typing import List, Optional, Tuple


def build_attention_and_labels(
    sequences,
    *,
    prompt_lens: List[int],
    eos_token_id: Optional[int],
    pad_token_id: Optional[int],
):
    """
    Build (attention_mask, labels, completion_lens) for CausalLM training.

    - Mask out prompt tokens and any padding beyond the first EOS.
    - completion_lens counts tokens after the prompt (at least 1 to avoid div-by-zero).
    """
    import torch

    if sequences.ndim != 2:
        raise ValueError("sequences must be [B, L]")
    bsz, seqlen = sequences.shape
    if len(prompt_lens) != bsz:
        raise ValueError("prompt_lens length mismatch")

    attention_mask = torch.zeros((bsz, seqlen), dtype=torch.long, device=sequences.device)
    labels = sequences.clone()
    completion_lens = torch.zeros((bsz,), dtype=torch.long, device=sequences.device)

    for i in range(bsz):
        p = int(prompt_lens[i])
        p = max(0, min(p, seqlen))

        end = seqlen
        seq = sequences[i]

        # Prefer first EOS after prompt as the natural end.
        if eos_token_id is not None:
            hits = (seq[p:] == int(eos_token_id)).nonzero(as_tuple=False)
            if hits.numel() > 0:
                end = min(end, p + int(hits[0].item()) + 1)  # include EOS token

        # If a real PAD token exists (and differs from EOS), trim at first PAD too.
        if pad_token_id is not None and pad_token_id != eos_token_id:
            hits = (seq[p:] == int(pad_token_id)).nonzero(as_tuple=False)
            if hits.numel() > 0:
                end = min(end, p + int(hits[0].item()))

        attention_mask[i, :end] = 1
        labels[i, :p] = -100
        labels[i, end:] = -100

        completion_lens[i] = max(1, end - p)

    return attention_mask, labels, completion_lens


def per_sample_nll(logits, labels) -> "torch.Tensor":
    """
    Compute per-sample NLL summed over non-ignored tokens (labels != -100).
    """
    import torch.nn.functional as F

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    vocab = shift_logits.size(-1)

    flat = F.cross_entropy(
        shift_logits.view(-1, vocab),
        shift_labels.view(-1),
        reduction="none",
        ignore_index=-100,
    )
    return flat.view(labels.size(0), -1).sum(dim=1)


def weighted_rl_loss(
    model,
    sequences,
    *,
    prompt_lens: List[int],
    weights: List[float],
    eos_token_id: Optional[int],
    pad_token_id: Optional[int],
    micro_batch_size: int = 0,
) -> Tuple["torch.Tensor", dict]:
    """
    Policy-gradient style loss for language models:

        loss = mean_i [ w_i * NLL_i / completion_len_i ]

    where w_i is a (possibly signed) advantage weight.
    """
    import torch

    attn, labels, comp_lens = build_attention_and_labels(
        sequences,
        prompt_lens=prompt_lens,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
    )

    bsz = int(sequences.size(0))
    if bsz == 0:
        raise ValueError("empty batch")

    # micro_batch_size <=0 means "no microbatching" (whole batch at once).
    mb = int(micro_batch_size) if micro_batch_size is not None else 0
    if mb <= 0 or mb > bsz:
        mb = bsz

    # Keep weights on device in the same dtype as NLL.
    w_full = None
    loss_sum = torch.zeros((), device=sequences.device, dtype=torch.float32)
    nll_mean_sum = torch.zeros((), device=sequences.device, dtype=torch.float32)
    w_sum = torch.zeros((), device=sequences.device, dtype=torch.float32)
    w_abs_sum = torch.zeros((), device=sequences.device, dtype=torch.float32)

    for start in range(0, bsz, mb):
        end = min(bsz, start + mb)
        seq_mb = sequences[start:end]
        attn_mb = attn[start:end]
        labels_mb = labels[start:end]
        comp_mb = comp_lens[start:end]

        out = model(input_ids=seq_mb, attention_mask=attn_mb, use_cache=False)
        nll_sum = per_sample_nll(out.logits, labels_mb)

        if w_full is None:
            w_full = torch.tensor(weights, device=sequences.device, dtype=nll_sum.dtype)
        w = w_full[start:end]

        nll_mean = nll_sum / comp_mb.to(nll_sum.dtype)
        loss_per = w * nll_mean

        loss_sum = loss_sum + loss_per.sum().to(loss_sum.dtype)
        nll_mean_sum = nll_mean_sum + nll_mean.sum().to(nll_mean_sum.dtype)
        w_sum = w_sum + w.sum().to(w_sum.dtype)
        w_abs_sum = w_abs_sum + w.abs().sum().to(w_abs_sum.dtype)

        # Free per-microbatch activations ASAP.
        del out

    # Keep loss as floating point (do NOT cast to sequences.dtype which is typically int64 token ids).
    loss = loss_sum / float(bsz)
    metrics = {
        "mean_nll": float((nll_mean_sum / float(bsz)).item()),
        "mean_weight": float((w_sum / float(bsz)).item()),
        "mean_abs_weight": float((w_abs_sum / float(bsz)).item()),
        "rl_microbatch": int(mb),
    }
    return loss, metrics
