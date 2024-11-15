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
conda env export > environment.yml
```

## Mesh to sdf

We use mesh-to-sdf that I forked because there was a bug

Remember to set `np.random.seed(42)` to always have the same SDF points.
