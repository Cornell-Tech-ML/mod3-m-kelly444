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
        
# Task 3.1: Parallelization & Task 3.2: Matrix Multiplication:
Diagnostics output from `python project/parallel_check.py`:
```
MAP
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_map.<locals>._map, 
/content/drive/MyDrive/mod3-m-kelly444/minitorch/fast_ops.py (182)  
================================================================================


Parallel loop listing for  Function tensor_map.<locals>._map, /content/drive/MyDrive/mod3-m-kelly444/minitorch/fast_ops.py (182) 
-------------------------------------------------------------------------|loop #ID
    def _map(                                                            | 
        out: Storage,                                                    | 
        out_shape: Shape,                                                | 
        out_strides: Strides,                                            | 
        in_storage: Storage,                                             | 
        in_shape: Shape,                                                 | 
        in_strides: Strides,                                             | 
    ) -> None:                                                           | 
        # TODO: Implement for Task 3.1.                                  | 
                                                                         | 
        if is_aligned(out_strides, in_strides, out_shape, in_shape):     | 
            # Directly iterate without indexing                          | 
            for i in prange(len(out)):-----------------------------------| #2
                out[i] = fn(in_storage[i])                               | 
            return                                                       | 
                                                                         | 
        # Parallel loop for mapping                                      | 
        for i in prange(len(out)):---------------------------------------| #3
            out_index: Index = np.zeros(MAX_DIMS, np.int32)--------------| #0
            in_index: Index = np.zeros(MAX_DIMS, np.int32)---------------| #1
            to_index(i, out_shape, out_index)                            | 
            broadcast_index(out_index, out_shape, in_shape, in_index)    | 
            out_pos = index_to_position(out_index, out_strides)          | 
            in_pos = index_to_position(in_index, in_strides)             | 
            out[out_pos] = fn(in_storage[in_pos])                        | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
 
Fused loop summary:
+--0 has the following loops fused into it:
   +--1 (fused)
Following the attempted fusion of parallel for-loops there are 3 parallel for-
loop(s) (originating from loops labelled: #2, #3, #0).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...
 
+--3 is a parallel loop
   +--0 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--3 (parallel)
   +--0 (parallel)
   +--1 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--3 (parallel)
   +--0 (serial, fused with loop(s): 1)


 
Parallel region 0 (loop #3) had 1 loop(s) fused and 1 loop(s) serialized as part
 of the larger parallel loop (#3).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at 
/content/drive/MyDrive/mod3-m-kelly444/minitorch/fast_ops.py (200) is hoisted 
out of the parallel loop labelled #3 (it will be performed before the loop is 
executed and reused inside the loop):
   Allocation:: out_index: Index = np.zeros(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/content/drive/MyDrive/mod3-m-kelly444/minitorch/fast_ops.py (201) is hoisted 
out of the parallel loop labelled #3 (it will be performed before the loop is 
executed and reused inside the loop):
   Allocation:: in_index: Index = np.zeros(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
None
ZIP
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_zip.<locals>._zip, 
/content/drive/MyDrive/mod3-m-kelly444/minitorch/fast_ops.py (235)  
================================================================================


Parallel loop listing for  Function tensor_zip.<locals>._zip, /content/drive/MyDrive/mod3-m-kelly444/minitorch/fast_ops.py (235) 
-------------------------------------------------------------------------------------|loop #ID
    def _zip(                                                                        | 
        out: Storage,                                                                | 
        out_shape: Shape,                                                            | 
        out_strides: Strides,                                                        | 
        a_storage: Storage,                                                          | 
        a_shape: Shape,                                                              | 
        a_strides: Strides,                                                          | 
        b_storage: Storage,                                                          | 
        b_shape: Shape,                                                              | 
        b_strides: Strides,                                                          | 
    ) -> None:                                                                       | 
        # TODO: Implement for Task 3.1.                                              | 
                                                                                     | 
        # if stride alignment                                                        | 
        if is_aligned(out_strides, a_strides, out_shape, a_shape) and is_aligned(    | 
            out_strides, b_strides, out_shape, b_shape                               | 
        ):                                                                           | 
            # Directly iterate without indexing                                      | 
            for i in prange(len(out)):-----------------------------------------------| #7
                out[i] = fn(a_storage[i], b_storage[i])                              | 
            return                                                                   | 
                                                                                     | 
        # Parallel loop for applying the function to each output element             | 
        for i in prange(len(out)):---------------------------------------------------| #8
            out_index: Index = np.zeros(MAX_DIMS, np.int32)--------------------------| #4
            a_index: Index = np.zeros(MAX_DIMS, np.int32)----------------------------| #5
            b_index: Index = np.zeros(MAX_DIMS, np.int32)----------------------------| #6
            to_index(i, out_shape, out_index)                                        | 
            out_pos = index_to_position(out_index, out_strides)                      | 
            broadcast_index(out_index, out_shape, a_shape, a_index)                  | 
            a_pos = index_to_position(a_index, a_strides)                            | 
            broadcast_index(out_index, out_shape, b_shape, b_index)                  | 
            b_pos = index_to_position(b_index, b_strides)                            | 
            out[out_pos] = fn(a_storage[a_pos], b_storage[b_pos])                    | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
 
Fused loop summary:
+--4 has the following loops fused into it:
   +--5 (fused)
   +--6 (fused)
Following the attempted fusion of parallel for-loops there are 3 parallel for-
loop(s) (originating from loops labelled: #7, #8, #4).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...
 
+--8 is a parallel loop
   +--4 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--8 (parallel)
   +--4 (parallel)
   +--5 (parallel)
   +--6 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--8 (parallel)
   +--4 (serial, fused with loop(s): 5, 6)


 
Parallel region 0 (loop #8) had 2 loop(s) fused and 1 loop(s) serialized as part
 of the larger parallel loop (#8).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at 
/content/drive/MyDrive/mod3-m-kelly444/minitorch/fast_ops.py (259) is hoisted 
out of the parallel loop labelled #8 (it will be performed before the loop is 
executed and reused inside the loop):
   Allocation:: out_index: Index = np.zeros(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/content/drive/MyDrive/mod3-m-kelly444/minitorch/fast_ops.py (260) is hoisted 
out of the parallel loop labelled #8 (it will be performed before the loop is 
executed and reused inside the loop):
   Allocation:: a_index: Index = np.zeros(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/content/drive/MyDrive/mod3-m-kelly444/minitorch/fast_ops.py (261) is hoisted 
out of the parallel loop labelled #8 (it will be performed before the loop is 
executed and reused inside the loop):
   Allocation:: b_index: Index = np.zeros(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
None
REDUCE
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_reduce.<locals>._reduce, 
/content/drive/MyDrive/mod3-m-kelly444/minitorch/fast_ops.py (294)  
================================================================================


Parallel loop listing for  Function tensor_reduce.<locals>._reduce, /content/drive/MyDrive/mod3-m-kelly444/minitorch/fast_ops.py (294) 
-------------------------------------------------------------------|loop #ID
    def _reduce(                                                   | 
        out: Storage,                                              | 
        out_shape: Shape,                                          | 
        out_strides: Strides,                                      | 
        a_storage: Storage,                                        | 
        a_shape: Shape,                                            | 
        a_strides: Strides,                                        | 
        reduce_dim: int,                                           | 
    ) -> None:                                                     | 
        # TODO: Implement for Task 3.1.                            | 
                                                                   | 
        reduce_size = a_shape[reduce_dim]                          | 
        for i in prange(len(out)):---------------------------------| #10
            out_index: Index = np.zeros(MAX_DIMS, np.int32)--------| #9
            to_index(i, out_shape, out_index)                      | 
            out_pos = index_to_position(out_index, out_strides)    | 
            out_val = out[out_pos]                                 | 
            for j in range(reduce_size):                           | 
                out_index[reduce_dim] = j                          | 
                a_pos = index_to_position(out_index, a_strides)    | 
                out_val = fn(out_val, a_storage[a_pos])            | 
            out[out_pos] = out_val                                 | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #10, #9).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...
 
+--10 is a parallel loop
   +--9 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--10 (parallel)
   +--9 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--10 (parallel)
   +--9 (serial)


 
Parallel region 0 (loop #10) had 0 loop(s) fused and 1 loop(s) serialized as 
part of the larger parallel loop (#10).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at 
/content/drive/MyDrive/mod3-m-kelly444/minitorch/fast_ops.py (307) is hoisted 
out of the parallel loop labelled #10 (it will be performed before the loop is 
executed and reused inside the loop):
   Allocation:: out_index: Index = np.zeros(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
None
MATRIX MULTIPLY
 
================================================================================
 Parallel Accelerator Optimizing:  Function _tensor_matrix_multiply, 
/content/drive/MyDrive/mod3-m-kelly444/minitorch/fast_ops.py (320)  
================================================================================


Parallel loop listing for  Function _tensor_matrix_multiply, /content/drive/MyDrive/mod3-m-kelly444/minitorch/fast_ops.py (320) 
--------------------------------------------------------------------------------------|loop #ID
def _tensor_matrix_multiply(                                                          | 
    out: Storage,                                                                     | 
    out_shape: Shape,                                                                 | 
    out_strides: Strides,                                                             | 
    a_storage: Storage,                                                               | 
    a_shape: Shape,                                                                   | 
    a_strides: Strides,                                                               | 
    b_storage: Storage,                                                               | 
    b_shape: Shape,                                                                   | 
    b_strides: Strides,                                                               | 
) -> None:                                                                            | 
    """NUMBA tensor matrix multiply function.                                         | 
                                                                                      | 
    Should work for any tensor shapes that broadcast as long as                       | 
                                                                                      | 
    ```                                                                               | 
    assert a_shape[-1] == b_shape[-2]                                                 | 
    ```                                                                               | 
                                                                                      | 
    Optimizations:                                                                    | 
                                                                                      | 
    * Outer loop in parallel                                                          | 
    * No index buffers or function calls                                              | 
    * Inner loop should have no global writes, 1 multiply.                            | 
                                                                                      | 
                                                                                      | 
    Args:                                                                             | 
    ----                                                                              | 
        out (Storage): storage for `out` tensor                                       | 
        out_shape (Shape): shape for `out` tensor                                     | 
        out_strides (Strides): strides for `out` tensor                               | 
        a_storage (Storage): storage for `a` tensor                                   | 
        a_shape (Shape): shape for `a` tensor                                         | 
        a_strides (Strides): strides for `a` tensor                                   | 
        b_storage (Storage): storage for `b` tensor                                   | 
        b_shape (Shape): shape for `b` tensor                                         | 
        b_strides (Strides): strides for `b` tensor                                   | 
                                                                                      | 
    Returns:                                                                          | 
    -------                                                                           | 
        None : Fills in `out`                                                         | 
                                                                                      | 
    """                                                                               | 
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0                            | 
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0                            | 
    assert a_shape[-1] == b_shape[-2]                                                 | 
                                                                                      | 
    for idx in prange(len(out)):------------------------------------------------------| #11
        # Calculate the position in the output storage                                | 
        batch = idx // (out_shape[-2] * out_shape[-1])                                | 
        i = (idx % (out_shape[-2] * out_shape[-1])) // out_shape[-1]                  | 
        j = idx % out_shape[-1]                                                       | 
        out_pos = batch * out_strides[0] + i * out_strides[1] + j * out_strides[2]    | 
        a_s = batch * a_batch_stride + i * a_strides[-2]                              | 
        b_s = batch * b_batch_stride + j * b_strides[-1]                              | 
        out_val = 0                                                                   | 
        # On out dimension                                                            | 
        for k in range(a_shape[-1]):                                                  | 
            a_pos = a_s + k * a_strides[-1]                                           | 
            b_pos = b_s + k * b_strides[-2]                                           | 
            out_val += a_storage[a_pos] * b_storage[b_pos]                            | 
        out[out_pos] = out_val                                                        | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #11).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None
```

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


