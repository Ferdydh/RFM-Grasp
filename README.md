# Project Documentation

## Project Structure

```
.
├── data/
├── src/
│   ├── core/
│   ├── data/
│   └── models/
├── scripts/            # Scripts here
│   ├── show_grasp.py
│   ├── sanity_mlp.py
│   └── sanity_transformer.py
└── pyproject.toml
```

## Getting Started

1. Place your dataset in the `data` directory at the project root
2. Use uv to install :
   ```bash
   uv venv
   uv pip install -e .
   ```

sudo apt-get install python3-opengl
sudo apt-get install libgl1-mesa-glx

if you open glb, you can open with glTF Model Viewer extension on vscode

Save with this`uv add your_package`

## Dataset Information

- There are 8836 files in the `acronym grasp` dataset.

- There are 7897 files in the `ShapeNetSem` dataset that follows the same format.

## Available Scripts

Run scripts using:

```bash
python -m scripts.script_name
```

Current scripts:

- `show_grasp.py`: Visualize dataset
- `sanity_mlp.py`: Test MLP model
- `sanity_transformer.py`: Test Transformer model

## Configuration System

The configuration system (defined in `src/core/config.py`) is a crucial component of this project. It provides a robust and type-safe way to configure all aspects of the experiments.

The default dataset split ratio is:

- Training: 90%
- Validation: 10%

### Quick Start Configuration

For quick experimentation, use:

```python
from src.core.config import MLPExperiment

config = MLPExperimentConfig.default()
```

For more detailed configuration options and parameters, refer to the docstrings in `src/core/config.py`.
