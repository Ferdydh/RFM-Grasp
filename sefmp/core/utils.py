import torch
import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(experiment_name: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    config_path = Path("sefmp") / Path("configs") / f"{experiment_name}.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
