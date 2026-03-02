import torch

def get_batch_seed(batch):
    """
    Extract the seed from the first item in the batch if it exists.
    """
    if "seed" in batch[0]:
        return batch[0]["seed"].item()
    return None

def get_batch_seed_collated(batch):
    """
    Extract the seed from the first item in the batch if it exists.
    """
    if "seed" in batch:
        return batch["seed"][0].item()
    return None

def seeded_rand(shape, device, seed=None):
    """
    Generate random numbers with an optional seed for reproducibility.
    """
    if seed is not None:
        g = torch.Generator(device=device)
        g.manual_seed(int(seed))
        return torch.rand(shape, generator=g, device=device)
    return torch.rand(shape, device=device)

def seeded_randint(high: int, size, device, seed=None):
    """
    Generate random integers with an optional seed for reproducibility.
    """
    if seed is not None:
        g = torch.Generator(device=device)
        g.manual_seed(int(seed))
        return torch.randint(low=0, high=high, size=size, generator=g, device=device)
    return torch.randint(low=0, high=high, size=size, device=device)