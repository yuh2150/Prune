"""
Configuration and initialization helpers for the evaluation module.
"""

import os
from pathlib import Path
from utils.general import set_logging, increment_path

def init_evaluation_env(project='runs/test', name='exp', exist_ok=False, save_txt=False):
    """
    Initialize evaluation environment, setup logging and create save directories.
    
    Args:
        project (str): Project directory.
        name (str): Experiment name.
        exist_ok (bool): Do not increment if directory exists.
        save_txt (bool): If True, create a 'labels' subdirectory.
        
    Returns:
        Path: Path to the save directory.
    """
    set_logging()
    save_dir = Path(increment_path(Path(project) / name, exist_ok=exist_ok))
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)
    return save_dir
