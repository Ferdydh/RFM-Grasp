## Hyperparameters

### Data Processing
- **Split Ratio (`split_ratio`)**: A list representing the data split between training, validation, and test sets in percentage.  
  Example: `[70, 15, 15]` for a 70% training, 15% validation, and 15% test split.

### Training Parameters
- **Device (`device`)**: The device used for PyTorch computations (e.g., `"cpu"`, `"cuda"`).  
  Default: `"cuda"` if a GPU is available, otherwise `"cpu"`.

- **Batch Size (`batch_size`)**: The number of samples processed per batch.  
  Default: `8`

- **Number of Workers (`num_workers`)**: Number of subprocesses used for data loading.  
  Default: `4`

## Usage
Instructions for setting or modifying these hyperparameters can be done in the configuration file or via a dictionary.

## Examples
- **Example 1**: Default split ratio `[80, 10, 10]`, device as `"cuda"`, batch size `8`, and `4` workers.
- **Example 2**: Custom split ratio `[70, 15, 15]`, device as `"cpu"`, batch size `16`, and `2` workers.