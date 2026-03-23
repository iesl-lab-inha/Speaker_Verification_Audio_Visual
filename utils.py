import torch


def lengths_to_mask(lengths, max_len=None):
    """
    lengths: [B]
    returns: [B, T] bool mask, True for valid positions
    """
    if max_len is None:
        max_len = int(lengths.max().item())
    rng = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    return rng < lengths.unsqueeze(1)
