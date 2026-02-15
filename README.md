# FM4PDE: Flow Matching for Partial Differential Equations

This is a codebase for solving Partial Differential Equations (PDEs) based on Flow Matching methods. It contains a complete pipeline for training and sampling PDE models, supporting various PDE types (e.g., Burgers, Darcy, Navier-Stokes, etc.).

## 📁 Project Structure

* **`configs/`**: Configuration files for different PDE experiments (YAML format).
* **`data/`**: Data generation and loading modules, including code related to `pdebench`.
* **`flow_matching/`**: Core implementation of Flow Matching, including ODE solvers, paths, and loss functions.
* **`models/`**: Model architecture definitions, primarily U-Net and its variants.
* **`output/`**: Storage path for training results.
* **`training/`**: Training loops, distributed training utilities, and model saving/loading logic.
* **`sampling/`**: Code related to sampling and generating PDE solutions.
* **`torchdiffeq/`**: Main implementation of Neural Ordinary Differential Equations (NODE).
* **`train.py`**: Main entry point script for training.
* **`train_arg_parser.py`**: The file defines the parameters that can be passed into the main training code.
* **`sample.py`**: Main entry point script for inference/sampling.
* **`run_train.sh`**: Example script for training.
* **`run_sample.sh`**: Example script for sampling.

## 🛠️ Installation & Environment

This project uses Conda for environment management. Please ensure you have Anaconda or Miniconda installed.

1.  **Create and activate the environment**:

    ```bash
    conda env create -f environment.yml
    conda activate fm4pde
    ```

2.  **Dependencies**: 
This code is primarily implemented in a Python 3.12 environment. To prevent version conflicts, it includes the implementation code of some libraries, such as `flow_matching`, `torchdiffeq`, and `pdebench`. Researchers can also visit the homepage of these libraries and follow the prompts to install the latest version to suit their tasks.

## 🚀 Usage

### 1. Training

`train.py` is the main entry point for neural network training, with arguments defined in `train_arg_parser.py`. Examples for single-node multi-GPU training from scratch are provided in `run_train.sh`. Additionally, training can be resumed from a checkpoint by passing the `--resume` argument. It should be noted that the example given in run_train.sh reads data and stores results in the default path. Please adjust the path according to your needs.

### 2. Sampling

After training, use `sample.py` to load the model and generate solutions. You can refer to `run_sample.sh` for usage examples. Default configurations for different PDEs can be found in the `configs/` directory. Supported PDE Types (inferred from filenames):

* Burger's Equation (`burger.yaml`)
* Darcy Flow (`darcy.yaml`)
* Helmholtz (`helmholtz.yaml`)
* Navier-Stokes (`nsnonbounded.yaml`)
* Poisson (`poisson.yaml`)
* Reaction-Diffusion (`reaction_diffusion.yaml`)
* Shallow Water (`shallow_water.yaml`)

Configurations in `sample.py` and the `configs/` directory must be tailored to the specific task. Users should pay particular attention to updating the paths for data input, model checkpoints, and result output to match their actual environment.

## 📝 Citation

If you use this code in your research, please cite the relevant paper (Add BibTeX here).

```

