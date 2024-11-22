# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

Run a bigger model and record the time per epoch reported by the trainer:

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html


You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```
python run_tests.py
```

* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py
        
# Task 3.1: Parallelization
Diagnostics output from `python project/parallel_check.py`:

# Task 3.2: Matrix Multiplication
Diagnostics output from `python project/parallel_check.py`:

# Task 3.3: CUDA Operations
# Task 3.4: CUDA Matrix Multiplication
Proving these lead to speed-ups on large matrix operations by making a graph comparing them to naive operations:
<img src="https://github.com/Cornell-Tech-ML/mod3-m-kelly444/blob/ad65ff51b24b68b0ee00ba9a4f53d4617373c23e/Screen%20Shot%202024-11-21%20at%204.18.21%20PM.png" width="50%">
# Task 3.5: Training
Results for training a tensor model and recording the time per epoch reported by the trainer + running a bigger model and recording the time per epoch reported by the trainer:

## Simple
* CPU
<img src="https://github.com/Cornell-Tech-ML/mod3-m-kelly444/blob/2ee058e96a84693d4313e3b75cb958b8c539e276/Screen%20Shot%202024-11-21%20at%206.34.14%20PM.png" width="50%">
<img src="https://github.com/Cornell-Tech-ML/mod3-m-kelly444/blob/19c5874cea49a3cd1a062d6b14d4e027d8be240f/Screen%20Shot%202024-11-21%20at%206.41.10%20PM.png" width="50%">

* GPU
<img src="https://github.com/Cornell-Tech-ML/mod3-m-kelly444/blob/8f14fbcfd6affb7689a28164b569213634d0ddf3/Screen%20Shot%202024-11-21%20at%2010.22.36%20PM.png" width="50%">
<img src="https://github.com/Cornell-Tech-ML/mod3-m-kelly444/blob/0b21ec718512ad509d14e933f39138070d45b0f8/Screen%20Shot%202024-11-21%20at%2010.23.00%20PM.png" width="50%">

* Bigger CPU
<img src="https://github.com/Cornell-Tech-ML/mod3-m-kelly444/blob/0b36c876316265e18b6d8f31bafb2a7ba7e2a158/Screen%20Shot%202024-11-21%20at%206.48.48%20PM.png" width="50%">
<img src="https://github.com/Cornell-Tech-ML/mod3-m-kelly444/blob/e83ce92e3be5dcd7c3d5f5819d615c6beaf445ea/Screen%20Shot%202024-11-21%20at%206.50.24%20PM.png" width="50%">

* Bigger GPU

## Split
* CPU
<img src="https://github.com/Cornell-Tech-ML/mod3-m-kelly444/blob/e1bb98139fecfeef3891dc991d7107cbd2a0ff50/Screen%20Shot%202024-11-21%20at%207.17.47%20PM.png" width="50%">
<img src="https://github.com/Cornell-Tech-ML/mod3-m-kelly444/blob/826c307f628f90c28d1d18cc94afe57f9f6579a4/Screen%20Shot%202024-11-21%20at%207.19.44%20PM.png" width="50%">

* GPU
* Bigger CPU
* Bigger GPU

## Xor
* CPU
* GPU
* Bigger CPU
* Bigger GPU

## Diag
* CPU
* GPU
* Bigger CPU
* Bigger GPU

## Circle
* CPU
* GPU
* Bigger CPU
* Bigger GPU

## Spiral
* CPU
* GPU
* Bigger CPU
* Bigger GPU


