import torch

def set_device() -> torch.device :

    """Sets the device we will use in the project depending on availability 

    Checks first if cuda/mps are available to use GPU, if not we set the device to CPU

    Args:
        None

    Returns:
        torch.device: The selected device."""

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    return device


def set_seeds(
        seed: int = 42,
        device: torch.device = 'cpu'
) -> None:
    
    """Sets random seeds for torch

    Args:
        seed (int, optional): random seed, default is 42
        device (torch.device, optional): the device that is used, CPU used by default 
    """

    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
    elif device.type == 'mps':
        torch.mps.manual_seed(seed)
    else:
        torch.manual_seed(seed)