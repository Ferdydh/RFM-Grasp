# Project Documentation

## Project Structure

```
.
├── data/          # Dataset directory
├── src/                 # Main source code
│   ├── core/           # Core functionality
│   ├── data/           # Data handling
│   └── models/         # Model implementations
├── scripts/            # Execution scripts
│   ├── show_grasp.py      # Data visualization
│   ├── sanity_mlp.py  # MLP model testing
│   └── sanity_transformer.py  # Transformer testing
└── environment.yaml    # Conda environment file
```

## Getting Started

1. Place your dataset in the `data` directory at the project root
2. Create and activate conda environment:
   ```bash
   conda env create -f environment.yaml
   conda activate adlr
   ```

Save with this
conda env export --no-builds > environment.yaml

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

- Training: 80%
- Validation: 10%
- Test: 10%

### Quick Start Configuration

For quick experimentation, use:

```python
from src.core.config import MLPExperiment

config = MLPExperimentConfig.default()
```

For more detailed configuration options and parameters, refer to the docstrings in `src/core/config.py`.
