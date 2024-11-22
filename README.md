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

# Task 3.5: Training
Results for training a tensor model and recording the time per epoch reported by the trainer + running a bigger model and recording the time per epoch reported by the trainer:

## Simple
* CPU
* GPU
* Bigger CPU
* Bigger GPU

## Split
* CPU
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
Bigger GPU

## Spiral
* CPU
* GPU
* Bigger CPU
* Bigger GPU


