# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

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

_____________________________________________________________________________________________

# Module 3 Homework Implementation

# 3.1 & 3.2 Parallel Diagnostics Output
```
MAP

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_map.<locals>._map,
/Users/maevekelly/workspace/mod3-m-kelly444 1.03.59 PM/minitorch/fast_ops.py
(142)
================================================================================


Parallel loop listing for  Function tensor_map.<locals>._map, /Users/maevekelly/workspace/mod3-m-kelly444 1.03.59 PM/minitorch/fast_ops.py (142)
-------------------------------------------------------------------------|loop #ID
    def _map(                                                            |
        out: Storage,                                                    |
        out_shape: Shape,                                                |
        out_strides: Strides,                                            |
        in_storage: Storage,                                             |
        in_shape: Shape,                                                 |
        in_strides: Strides,                                             |
    ) -> None:                                                           |
        for i in prange(len(out)):---------------------------------------| #0
            # Thread-safe index computation                              |
            out_index = np.empty(MAX_DIMS, np.int32)                     |
            in_index = np.empty(MAX_DIMS, np.int32)                      |
                                                                         |
            to_index(i, out_shape, out_index)                            |
            broadcast_index(out_index, out_shape, in_shape, in_index)    |
            o = index_to_position(out_index, out_strides)                |
            j = index_to_position(in_index, in_strides)                  |
            out[o] = fn(in_storage[j])                                   |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #0).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at
/Users/maevekelly/workspace/mod3-m-kelly444 1.03.59 PM/minitorch/fast_ops.py
(152) is hoisted out of the parallel loop labelled #0 (it will be performed
before the loop is executed and reused inside the loop):
   Allocation:: out_index = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
/Users/maevekelly/workspace/mod3-m-kelly444 1.03.59 PM/minitorch/fast_ops.py
(153) is hoisted out of the parallel loop labelled #0 (it will be performed
before the loop is executed and reused inside the loop):
   Allocation:: in_index = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
None
ZIP

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_zip.<locals>._zip,
/Users/maevekelly/workspace/mod3-m-kelly444 1.03.59 PM/minitorch/fast_ops.py
(179)
================================================================================


Parallel loop listing for  Function tensor_zip.<locals>._zip, /Users/maevekelly/workspace/mod3-m-kelly444 1.03.59 PM/minitorch/fast_ops.py (179)
-----------------------------------------------------------------------|loop #ID
    def _zip(                                                          |
        out: Storage,                                                  |
        out_shape: Shape,                                              |
        out_strides: Strides,                                          |
        a_storage: Storage,                                            |
        a_shape: Shape,                                                |
        a_strides: Strides,                                            |
        b_storage: Storage,                                            |
        b_shape: Shape,                                                |
        b_strides: Strides,                                            |
    ) -> None:                                                         |
        for i in prange(len(out)):-------------------------------------| #1
            # Per-thread index buffers                                 |
            out_index = np.empty(MAX_DIMS, np.int32)                   |
            a_index = np.empty(MAX_DIMS, np.int32)                     |
            b_index = np.empty(MAX_DIMS, np.int32)                     |
                                                                       |
            to_index(i, out_shape, out_index)                          |
            o = index_to_position(out_index, out_strides)              |
            broadcast_index(out_index, out_shape, a_shape, a_index)    |
            j = index_to_position(a_index, a_strides)                  |
            broadcast_index(out_index, out_shape, b_shape, b_index)    |
            k = index_to_position(b_index, b_strides)                  |
            out[o] = fn(a_storage[j], b_storage[k])                    |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #1).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at
/Users/maevekelly/workspace/mod3-m-kelly444 1.03.59 PM/minitorch/fast_ops.py
(192) is hoisted out of the parallel loop labelled #1 (it will be performed
before the loop is executed and reused inside the loop):
   Allocation:: out_index = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
/Users/maevekelly/workspace/mod3-m-kelly444 1.03.59 PM/minitorch/fast_ops.py
(193) is hoisted out of the parallel loop labelled #1 (it will be performed
before the loop is executed and reused inside the loop):
   Allocation:: a_index = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
/Users/maevekelly/workspace/mod3-m-kelly444 1.03.59 PM/minitorch/fast_ops.py
(194) is hoisted out of the parallel loop labelled #1 (it will be performed
before the loop is executed and reused inside the loop):
   Allocation:: b_index = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
None
REDUCE

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_reduce.<locals>._reduce,
/Users/maevekelly/workspace/mod3-m-kelly444 1.03.59 PM/minitorch/fast_ops.py
(220)
================================================================================


Parallel loop listing for  Function tensor_reduce.<locals>._reduce, /Users/maevekelly/workspace/mod3-m-kelly444 1.03.59 PM/minitorch/fast_ops.py (220)
-----------------------------------------------------------------|loop #ID
    def _reduce(                                                 |
        out: Storage,                                            |
        out_shape: Shape,                                        |
        out_strides: Strides,                                    |
        a_storage: Storage,                                      |
        a_shape: Shape,                                          |
        a_strides: Strides,                                      |
        reduce_dim: int,                                         |
    ) -> None:                                                   |
        out_index = np.zeros(MAX_DIMS, np.int32)-----------------| #2
        reduce_size = a_shape[reduce_dim]                        |
                                                                 |
        for i in prange(len(out)):-------------------------------| #3
            # Per-thread index management                        |
            out_index = np.empty(MAX_DIMS, np.int32)             |
            local_index = np.empty(MAX_DIMS, np.int32)           |
                                                                 |
            to_index(i, out_shape, out_index)                    |
            o = index_to_position(out_index, out_strides)        |
                                                                 |
            for j in range(len(out_shape)):                      |
                local_index[j] = out_index[j]                    |
                                                                 |
            # Reduction loop                                     |
            for s in range(reduce_size):                         |
                local_index[reduce_dim] = s                      |
                j = index_to_position(local_index, a_strides)    |
                out[o] = fn(out[o], a_storage[j])                |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #3).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at
/Users/maevekelly/workspace/mod3-m-kelly444 1.03.59 PM/minitorch/fast_ops.py
(234) is hoisted out of the parallel loop labelled #3 (it will be performed
before the loop is executed and reused inside the loop):
   Allocation:: out_index = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
/Users/maevekelly/workspace/mod3-m-kelly444 1.03.59 PM/minitorch/fast_ops.py
(235) is hoisted out of the parallel loop labelled #3 (it will be performed
before the loop is executed and reused inside the loop):
   Allocation:: local_index = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
None
MATRIX MULTIPLY

================================================================================
 Parallel Accelerator Optimizing:  Function _tensor_matrix_multiply,
/Users/maevekelly/workspace/mod3-m-kelly444 1.03.59 PM/minitorch/fast_ops.py
(252)
================================================================================


Parallel loop listing for  Function _tensor_matrix_multiply, /Users/maevekelly/workspace/mod3-m-kelly444 1.03.59 PM/minitorch/fast_ops.py (252)
---------------------------------------------------------------------------|loop #ID
def _tensor_matrix_multiply(                                               |
    out: Storage,                                                          |
    out_shape: Shape,                                                      |
    out_strides: Strides,                                                  |
    a_storage: Storage,                                                    |
    a_shape: Shape,                                                        |
    a_strides: Strides,                                                    |
    b_storage: Storage,                                                    |
    b_shape: Shape,                                                        |
    b_strides: Strides,                                                    |
) -> None:                                                                 |
    """Implements efficient batched matrix multiplication for tensors.     |
                                                                           |
    Performance optimizations:                                             |
    1. Parallel processing of output rows                                  |
    2. Minimized memory access patterns                                    |
    3. Cache-friendly inner loop                                           |
    4. Direct stride manipulation                                          |
                                                                           |
    The implementation handles:                                            |
    - Batched operations with broadcasting                                 |
    - Efficient memory access patterns                                     |
    - Parallel computation across output elements                          |
    """                                                                    |
    # Handle batch dimension strides                                       |
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0                 |
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0                 |
    blocks = a_shape[-1]                                                   |
                                                                           |
    # Parallel processing of output matrix                                 |
    for row_i in prange(0, out_shape[0]):----------------------------------| #4
        for col_j in range(0, out_shape[1]):                               |
            for block_k in range(0, out_shape[2]):                         |
                # Compute base positions                                   |
                row_s = row_i * a_batch_stride + col_j * a_strides[1]      |
                col_s = row_i * b_batch_stride + block_k * b_strides[2]    |
                                                                           |
                temp = 0.0                                                 |
                                                                           |
                # Matrix multiplication inner loop                         |
                for _ in range(0, blocks):                                 |
                    temp += a_storage[row_s] * b_storage[col_s]            |
                    row_s += a_strides[-1]                                 |
                    col_s += b_strides[-2]                                 |
                                                                           |
                # Store result                                             |
                out[                                                       |
                    row_i * out_strides[0]                                 |
                    + col_j * out_strides[1]                               |
                    + block_k * out_strides[2]                             |
                ] = temp                                                   |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #4).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

```