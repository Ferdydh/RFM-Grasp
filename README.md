There are 8836 files in the `acronym grasp` dataset.

There are 7897 files in the `ShapeNetSem` dataset that follows the same format.

# Dev

## workflow

Setup conda with

```bash
conda env create -f environment.yml
```

Save your new conda env (if you installed something with conda install or pip install)

```bash
conda env export --no-builds > environment.yaml
```

## Mesh to sdf

We use mesh2sdf
