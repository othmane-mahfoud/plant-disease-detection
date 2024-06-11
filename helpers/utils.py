import os
import torch

from pathlib import Path
from torch.utils.tensorboard.writer import SummaryWriter
from datetime import datetime
from typing import Tuple

def create_writer(experiment_name: str, 
                  model_name: str, 
                  extra: str=None) -> Tuple[torch.utils.tensorboard.writer.SummaryWriter, torch.utils.tensorboard.writer.SummaryWriter]:
    """Creates torch.utils.tensorboard.writer.SummaryWriter() instances saving to specific log_dirs.

    log_dirs are combinations of runs/timestamp/experiment_name/model_name/extra.

    Where timestamp is the current date in YYYY-MM-DD format.

    Args:
        experiment_name (str): Name of experiment.
        model_name (str): Name of model.
        extra (str, optional): Anything extra to add to the directory. Defaults to None.

    Returns:
        Tuple[torch.utils.tensorboard.writer.SummaryWriter, torch.utils.tensorboard.writer.SummaryWriter]:
            Instances of writers saving to local and Google Drive log_dirs.

    Example usage:
        # Create writers saving to local and Google Drive directories
        writer_local, writer_drive = create_writer(experiment_name="data_10_percent",
                                                   model_name="effnetb2",
                                                   extra="5_epochs")
    """
    # Get timestamp of current date (all experiments on certain day live in same folder)
    timestamp = datetime.now().strftime("%Y-%m-%d") # returns current date in YYYY-MM-DD format

    if extra:
        # Create log directory paths
        log_dir_local = os.path.join("runs", timestamp, experiment_name, model_name, extra)
        log_dir_drive = os.path.join("/content/drive/MyDrive/plant-disease-detection/runs", timestamp, experiment_name, model_name, extra)
    else:
        log_dir_local = os.path.join("runs", timestamp, experiment_name, model_name)
        log_dir_drive = os.path.join("/content/drive/MyDrive/plant-disease-detection/runs", timestamp, experiment_name, model_name)
        
    print(f"[INFO] Created SummaryWriters, saving to: {log_dir_local} and {log_dir_drive}...")
    return SummaryWriter(log_dir=log_dir_local), SummaryWriter(log_dir=log_dir_drive)



def save_model(model: torch.nn.Module,
    target_dir: str,
    model_name: str):
    """Saves model to specified directories (locally and on Google Drive).

    Args:
    model: the PyTorch model to save
    target_dir: the directory where you want to save it
    model_name: the name under which you want your model to be saved must end in ".pth"
    """
    # Local directory
    target_dir_path_local = Path(target_dir)
    target_dir_path_local.mkdir(parents=True, exist_ok=True)
    save_path_local = target_dir_path_local / model_name

    # Google Drive directory
    target_dir_path_drive = Path(f"/content/drive/MyDrive/plant-disease-detection/{target_dir}")
    target_dir_path_drive.mkdir(parents=True, exist_ok=True)
    save_path_drive = target_dir_path_drive / model_name

    # Save the model state_dict() to both locations
    torch.save(obj=model.state_dict(), f=save_path_local)
    torch.save(obj=model.state_dict(), f=save_path_drive)
    
    print(f"[INFO] Model saved to: {save_path_local} and {save_path_drive}")
