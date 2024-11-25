There are 8836 files in the `acronym grasp` dataset.

There are 7897 files in the `ShapeNetSem` dataset that follows the same format.

# Dev

## workflow

Setup conda with

```bash
conda env create -f environment.yaml
```

Save your new conda env (if you installed something with conda install or pip install)

```bash
conda env export --no-builds > environment.yaml
```

## How to run

1. Install the conda environment and turn it on
2. on root: `python sefmp/main.py --help`
