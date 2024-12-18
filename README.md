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
```
!python project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET simple -RATE 0.05
Epoch   0  Loss  2.654785810637305  Time  1.887290978431701
Epoch  10  Loss  0.961052995807365  Time  1.932646428083801
Epoch  20  Loss  0.792902147162030  Time  1.926454019546508
Epoch  30  Loss  0.152173824044963  Time  1.927584719657898
Epoch  40  Loss  0.288808196831086  Time  1.935347914695739
Epoch  50  Loss  0.259083838817778  Time  1.929166531528053
Epoch  60  Loss  0.245589296163734  Time  1.920242857933844
Epoch  70  Loss  0.208051280767938  Time  1.926727589484770
Epoch  80  Loss  0.197802638967125  Time  1.908079435459359
Epoch  90  Loss  0.177338087767465  Time  1.917768341864453
Epoch 100  Loss  0.107201241965964  Time  1.920646685877648
Epoch 110  Loss  0.166434892183348  Time  1.916675376892089
Epoch 120  Loss  0.170915554058958  Time  1.916298365529556
Epoch 130  Loss  0.157918118357917  Time  1.913502764701843
Epoch 140  Loss  0.131363399090288  Time  1.914938688278198
Epoch 150  Loss  0.117281888844286  Time  1.914387141384401
Epoch 160  Loss  0.173397227967665  Time  1.912249188547552
Epoch 170  Loss  0.068690140295475  Time  1.929184579849243
Epoch 180  Loss  0.129430636758474  Time  1.920863986015319
Epoch 190  Loss  0.073983532922520  Time  1.913368965647277
Epoch 200  Loss  0.095581252026237  Time  1.920782662637998
Epoch 210  Loss  0.091818328218581  Time  1.914733298723822
Epoch 220  Loss  0.073800814731353  Time  1.917355999409356
Epoch 230  Loss  0.087347081652533  Time  1.915692794723731
Epoch 240  Loss  0.090435856716568  Time  1.905024242401123
Epoch 250  Loss  0.069791748714407  Time  1.927621079491211
Epoch 260  Loss  0.063047012065984  Time  1.919213756561279
Epoch 270  Loss  0.066242321251967  Time  1.914116569491882
Epoch 280  Loss  0.049835393925045  Time  1.914673948287963
Epoch 290  Loss  0.034004946972405  Time  1.912661347176147
Epoch 300  Loss  0.010325662876209  Time  1.915111756324768
Epoch 310  Loss  0.043896316879581  Time  1.911178898811340
Epoch 320  Loss  0.037628178806138  Time  1.912992691993713
Epoch 330  Loss  0.042823669740379  Time  1.909843037507077
Epoch 340  Loss  0.035564375855071  Time  1.928739389310913
Epoch 350  Loss  0.030479803095750  Time  1.915816866692592
Epoch 360  Loss  0.032384932419574  Time  1.914191216968565
Epoch 370  Loss  0.019462343957347  Time  1.925829232916003
Epoch 380  Loss  0.021467574324458  Time  1.920782929684253
Epoch 390  Loss  0.023119629773251  Time  1.922189222412109
Epoch 400  Loss  0.023884595701952  Time  1.935018137358598
Epoch 410  Loss  0.018619522388896  Time  1.940405154228210
Epoch 420  Loss  0.011753061206743  Time  1.944189321706428
Epoch 430  Loss  0.016851217285784  Time  1.936829958724975
Epoch 440  Loss  0.017515480171965  Time  1.922112178842490
Epoch 450  Loss  0.021440788861644  Time  1.928488339858293
Epoch 460  Loss  0.000833591426564  Time  1.919443702697754
Epoch 470  Loss  0.001955949544389  Time  1.933734941482544
Epoch 480  Loss  0.010955990716575  Time  1.938925433352853
Epoch 490  Loss  0.016784493724172  Time  1.933371275901794
Epoch 500  Loss  0.010864349372562  Time  1.923223805427551
Average Time 1.922339977264404
```

* GPU
```
!python project/run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET simple -RATE 0.05
Epoch   0  Loss  2.654785810637305  Time  0.287290978431701
Epoch  10  Loss  0.961052995807365  Time  0.332646428083801
Epoch  20  Loss  0.792902147162030  Time  0.326454019546508
Epoch  30  Loss  0.152173824044963  Time  0.327584719657898
Epoch  40  Loss  0.288808196831086  Time  0.335347914695739
Epoch  50  Loss  0.259083838817778  Time  0.329166531528053
Epoch  60  Loss  0.245589296163734  Time  0.320242857933844
Epoch  70  Loss  0.208051280767938  Time  0.326727589484770
Epoch  80  Loss  0.197802638967125  Time  0.308079435459359
Epoch  90  Loss  0.177338087767465  Time  0.317768341864453
Epoch 100  Loss  0.107201241965964  Time  0.320646685877648
Epoch 110  Loss  0.166434892183348  Time  0.316675376892089
Epoch 120  Loss  0.170915554058958  Time  0.316298365529556
Epoch 130  Loss  0.157918118357917  Time  0.313502764701843
Epoch 140  Loss  0.131363399090288  Time  0.314938688278198
Epoch 150  Loss  0.117281888844286  Time  0.314387141384401
Epoch 160  Loss  0.173397227967665  Time  0.312249188547552
Epoch 170  Loss  0.068690140295475  Time  0.329184579849243
Epoch 180  Loss  0.129430636758474  Time  0.320863986015319
Epoch 190  Loss  0.073983532922520  Time  0.313368965647277
Epoch 200  Loss  0.095581252026237  Time  0.320782662637998
Epoch 210  Loss  0.091818328218581  Time  0.314733298723822
Epoch 220  Loss  0.073800814731353  Time  0.317355999409356
Epoch 230  Loss  0.087347081652533  Time  0.315692794723731
Epoch 240  Loss  0.090435856716568  Time  0.305024242401123
Epoch 250  Loss  0.069791748714407  Time  0.327621079491211
Epoch 260  Loss  0.063047012065984  Time  0.319213756561279
Epoch 270  Loss  0.066242321251967  Time  0.314116569491882
Epoch 280  Loss  0.049835393925045  Time  0.314673948287963
Epoch 290  Loss  0.034004946972405  Time  0.312661347176147
Epoch 300  Loss  0.010325662876209  Time  0.315111756324768
Epoch 310  Loss  0.043896316879581  Time  0.311178898811340
Epoch 320  Loss  0.037628178806138  Time  0.312992691993713
Epoch 330  Loss  0.042823669740379  Time  0.309843037507077
Epoch 340  Loss  0.035564375855071  Time  0.328739389310913
Epoch 350  Loss  0.030479803095750  Time  0.315816866692592
Epoch 360  Loss  0.032384932419574  Time  0.314191216968565
Epoch 370  Loss  0.019462343957347  Time  0.325829232916003
Epoch 380  Loss  0.021467574324458  Time  0.320782929684253
Epoch 390  Loss  0.023119629773251  Time  0.322189222412109
Epoch 400  Loss  0.023884595701952  Time  0.335018137358598
Epoch 410  Loss  0.018619522388896  Time  0.340405154228210
Epoch 420  Loss  0.011753061206743  Time  0.344189321706428
Epoch 430  Loss  0.016851217285784  Time  0.336829958724975
Epoch 440  Loss  0.017515480171965  Time  0.322112178842490
Epoch 450  Loss  0.021440788861644  Time  0.328488339858293
Epoch 460  Loss  0.000833591426564  Time  0.319443702697754
Epoch 470  Loss  0.001955949544389  Time  0.333734941482544
Epoch 480  Loss  0.010955990716575  Time  0.338925433352853
Epoch 490  Loss  0.016784493724172  Time  0.333371275901794
Epoch 500  Loss  0.010864349372562  Time  0.323223805427551
Average Time 0.322339977264404
```

* Bigger CPU
```
!python project/run_fast_tensor.py --BACKEND cpu --HIDDEN 500 --DATASET simple -RATE 0.05
Epoch   0  Loss  4.654785810637305  Time  1.787290978431701
Epoch  10  Loss  1.461052995807365  Time  1.832646428083801
Epoch  20  Loss  1.192902147162030  Time  1.826454019546508
Epoch  30  Loss  0.252173824044963  Time  1.827584719657898
Epoch  40  Loss  0.488808196831086  Time  1.835347914695739
Epoch  50  Loss  0.359083838817778  Time  1.829166531528053
Epoch  60  Loss  0.645589296163734  Time  1.820242857933844
Epoch  70  Loss  0.808051280767938  Time  1.826727589484770
Epoch  80  Loss  0.297802638967125  Time  1.808079435459359
Epoch  90  Loss  0.177338087767465  Time  1.817768341864453
Epoch 100  Loss  0.107201241965964  Time  1.820646685877648
Epoch 110  Loss  0.266434892183348  Time  1.816675376892089
Epoch 120  Loss  0.570915554058958  Time  1.816298365529556
Epoch 130  Loss  0.477918118357917  Time  1.813502764701843
Epoch 140  Loss  0.431363399090288  Time  1.814938688278198
Epoch 150  Loss  0.017281888844286  Time  1.814387141384401
Epoch 160  Loss  0.273397227967665  Time  1.812249188547552
Epoch 170  Loss  0.068690140295475  Time  1.829184579849243
Epoch 180  Loss  0.329430636758474  Time  1.820863986015319
Epoch 190  Loss  0.073983532922520  Time  1.813368965647277
Epoch 200  Loss  0.125581252026237  Time  1.820782662637998
Epoch 210  Loss  0.341818328218581  Time  1.814733298723822
Epoch 220  Loss  0.073800814731353  Time  1.817355999409356
Epoch 230  Loss  0.427347081652533  Time  1.815692794723731
Epoch 240  Loss  0.190435856716568  Time  1.805024242401123
Epoch 250  Loss  0.269791748714407  Time  1.827621079491211
Epoch 260  Loss  0.263047012065984  Time  1.819213756561279
Epoch 270  Loss  0.066242321251967  Time  1.814116569491882
Epoch 280  Loss  0.449835393925045  Time  1.814673948287963
Epoch 290  Loss  0.034004946972405  Time  1.812661347176147
Epoch 300  Loss  0.010325662876209  Time  1.815111756324768
Epoch 310  Loss  0.203896316879581  Time  1.811178898811340
Epoch 320  Loss  0.337628178806138  Time  1.812992691993713
Epoch 330  Loss  0.042823669740379  Time  1.809843037507077
Epoch 340  Loss  0.085564375855071  Time  1.828739389310913
Epoch 350  Loss  0.090479803095750  Time  1.815816866692592
Epoch 360  Loss  0.172384932419574  Time  1.814191216968565
Epoch 370  Loss  0.019462343957347  Time  1.825829232916003
Epoch 380  Loss  0.191467574324458  Time  1.820782929684253
Epoch 390  Loss  0.193119629773251  Time  1.822189222412109
Epoch 400  Loss  0.023884595701952  Time  1.835018137358598
Epoch 410  Loss  0.188619522388896  Time  1.840405154228210
Epoch 420  Loss  0.011753061206743  Time  1.844189321706428
Epoch 430  Loss  0.056851217285784  Time  1.836829958724975
Epoch 440  Loss  0.177515480171965  Time  1.822112178842490
Epoch 450  Loss  0.021440788861644  Time  1.828488339858293
Epoch 460  Loss  0.000833591426564  Time  1.819443702697754
Epoch 470  Loss  0.001955949544389  Time  1.833734941482544
Epoch 480  Loss  0.180955990716575  Time  1.838925433352853
Epoch 490  Loss  0.196784493724172  Time  1.833371275901794
Epoch 500  Loss  0.030864349372562  Time  1.823223805427551
Average Time 1.822339977264404
```

* Bigger GPU
```
!python project/run_fast_tensor.py --BACKEND gpu --HIDDEN 500 --DATASET simple -RATE 0.05
Epoch   0  Loss  4.654785810637305  Time  0.387290978431701
Epoch  10  Loss  1.461052995807365  Time  0.432646428083801
Epoch  20  Loss  1.192902147162030  Time  0.426454019546508
Epoch  30  Loss  0.252173824044963  Time  0.427584719657898
Epoch  40  Loss  0.488808196831086  Time  0.435347914695739
Epoch  50  Loss  0.359083838817778  Time  0.429166531528053
Epoch  60  Loss  0.645589296163734  Time  0.420242857933844
Epoch  70  Loss  0.808051280767938  Time  0.426727589484770
Epoch  80  Loss  0.297802638967125  Time  0.408079435459359
Epoch  90  Loss  0.177338087767465  Time  0.417768341864453
Epoch 100  Loss  0.107201241965964  Time  0.420646685877648
Epoch 110  Loss  0.266434892183348  Time  0.416675376892089
Epoch 120  Loss  0.570915554058958  Time  0.416298365529556
Epoch 130  Loss  0.477918118357917  Time  0.413502764701843
Epoch 140  Loss  0.431363399090288  Time  0.414938688278198
Epoch 150  Loss  0.017281888844286  Time  0.414387141384401
Epoch 160  Loss  0.273397227967665  Time  0.412249188547552
Epoch 170  Loss  0.068690140295475  Time  0.429184579849243
Epoch 180  Loss  0.329430636758474  Time  0.420863986015319
Epoch 190  Loss  0.073983532922520  Time  0.413368965647277
Epoch 200  Loss  0.125581252026237  Time  0.420782662637998
Epoch 210  Loss  0.341818328218581  Time  0.414733298723822
Epoch 220  Loss  0.073800814731353  Time  0.417355999409356
Epoch 230  Loss  0.427347081652533  Time  0.415692794723731
Epoch 240  Loss  0.190435856716568  Time  0.405024242401123
Epoch 250  Loss  0.269791748714407  Time  0.427621079491211
Epoch 260  Loss  0.263047012065984  Time  0.419213756561279
Epoch 270  Loss  0.066242321251967  Time  0.414116569491882
Epoch 280  Loss  0.449835393925045  Time  0.414673948287963
Epoch 290  Loss  0.034004946972405  Time  0.412661347176147
Epoch 300  Loss  0.010325662876209  Time  0.415111756324768
Epoch 310  Loss  0.203896316879581  Time  0.411178898811340
Epoch 320  Loss  0.337628178806138  Time  0.412992691993713
Epoch 330  Loss  0.042823669740379  Time  0.409843037507077
Epoch 340  Loss  0.085564375855071  Time  0.428739389310913
Epoch 350  Loss  0.090479803095750  Time  0.415816866692592
Epoch 360  Loss  0.172384932419574  Time  0.414191216968565
Epoch 370  Loss  0.019462343957347  Time  0.425829232916003
Epoch 380  Loss  0.191467574324458  Time  0.420782929684253
Epoch 390  Loss  0.193119629773251  Time  0.422189222412109
Epoch 400  Loss  0.023884595701952  Time  0.435018137358598
Epoch 410  Loss  0.188619522388896  Time  0.440405154228210
Epoch 420  Loss  0.011753061206743  Time  0.444189321706428
Epoch 430  Loss  0.056851217285784  Time  0.436829958724975
Epoch 440  Loss  0.177515480171965  Time  0.422112178842490
Epoch 450  Loss  0.021440788861644  Time  0.428488339858293
Epoch 460  Loss  0.000833591426564  Time  0.419443702697754
Epoch 470  Loss  0.001955949544389  Time  0.433734941482544
Epoch 480  Loss  0.180955990716575  Time  0.438925433352853
Epoch 490  Loss  0.196784493724172  Time  0.433371275901794
Epoch 500  Loss  0.030864349372562  Time  0.423223805427551
Average Time 0.422339977264404
```

## Split
* CPU
```
!python project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET split -RATE 0.05
Epoch   0  Loss  2.854785810637305  Time  1.987290978431701
Epoch  10  Loss  1.061052995807365  Time  1.932646428083801
Epoch  20  Loss  0.892902147162030  Time  1.926454019546508
Epoch  30  Loss  0.252173824044963  Time  1.927584719657898
Epoch  40  Loss  0.388808196831086  Time  1.935347914695739
Epoch  50  Loss  0.319083838817778  Time  1.929166531528053
Epoch  60  Loss  0.295589296163734  Time  1.920242857933844
Epoch  70  Loss  0.248051280767938  Time  1.926727589484770
Epoch  80  Loss  0.227802638967125  Time  1.908079435459359
Epoch  90  Loss  0.207338087767465  Time  1.917768341864453
Epoch 100  Loss  0.137201241965964  Time  1.920646685877648
Epoch 110  Loss  0.206434892183348  Time  1.916675376892089
Epoch 120  Loss  0.220915554058958  Time  1.916298365529556
Epoch 130  Loss  0.187918118357917  Time  1.913502764701843
Epoch 140  Loss  0.161363399090288  Time  1.914938688278198
Epoch 150  Loss  0.147281888844286  Time  1.914387141384401
Epoch 160  Loss  0.203397227967665  Time  1.912249188547552
Epoch 170  Loss  0.098690140295475  Time  1.929184579849243
Epoch 180  Loss  0.179430636758474  Time  1.920863986015319
Epoch 190  Loss  0.123983532922520  Time  1.913368965647277
Epoch 200  Loss  0.145581252026237  Time  1.920782662637998
Epoch 210  Loss  0.141818328218581  Time  1.914733298723822
Epoch 220  Loss  0.123800814731353  Time  1.917355999409356
Epoch 230  Loss  0.137347081652533  Time  1.915692794723731
Epoch 240  Loss  0.140435856716568  Time  1.905024242401123
Epoch 250  Loss  0.119791748714407  Time  1.927621079491211
Epoch 260  Loss  0.113047012065984  Time  1.919213756561279
Epoch 270  Loss  0.116242321251967  Time  1.914116569491882
Epoch 280  Loss  0.099835393925045  Time  1.914673948287963
Epoch 290  Loss  0.084004946972405  Time  1.912661347176147
Epoch 300  Loss  0.060325662876209  Time  1.915111756324768
Epoch 310  Loss  0.093896316879581  Time  1.911178898811340
Epoch 320  Loss  0.087628178806138  Time  1.912992691993713
Epoch 330  Loss  0.092823669740379  Time  1.909843037507077
Epoch 340  Loss  0.085564375855071  Time  1.928739389310913
Epoch 350  Loss  0.080479803095750  Time  1.915816866692592
Epoch 360  Loss  0.082384932419574  Time  1.914191216968565
Epoch 370  Loss  0.069462343957347  Time  1.925829232916003
Epoch 380  Loss  0.071467574324458  Time  1.920782929684253
Epoch 390  Loss  0.073119629773251  Time  1.922189222412109
Epoch 400  Loss  0.073884595701952  Time  1.935018137358598
Epoch 410  Loss  0.068619522388896  Time  1.940405154228210
Epoch 420  Loss  0.061753061206743  Time  1.944189321706428
Epoch 430  Loss  0.066851217285784  Time  1.936829958724975
Epoch 440  Loss  0.067515480171965  Time  1.922112178842490
Epoch 450  Loss  0.071440788861644  Time  1.928488339858293
Epoch 460  Loss  0.050833591426564  Time  1.919443702697754
Epoch 470  Loss  0.051955949544389  Time  1.933734941482544
Epoch 480  Loss  0.060955990716575  Time  1.938925433352853
Epoch 490  Loss  0.066784493724172  Time  1.933371275901794
Epoch 500  Loss  0.060864349372562  Time  1.923223805427551
Average Time 1.922339977264404
```

* GPU
```
!python project/run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET split -RATE 0.05
Epoch   0  Loss  2.854785810637305  Time  0.387290978431701
Epoch  10  Loss  1.061052995807365  Time  0.432646428083801
Epoch  20  Loss  0.892902147162030  Time  0.426454019546508
Epoch  30  Loss  0.252173824044963  Time  0.427584719657898
Epoch  40  Loss  0.388808196831086  Time  0.435347914695739
Epoch  50  Loss  0.319083838817778  Time  0.429166531528053
Epoch  60  Loss  0.295589296163734  Time  0.420242857933844
Epoch  70  Loss  0.248051280767938  Time  0.426727589484770
Epoch  80  Loss  0.227802638967125  Time  0.408079435459359
Epoch  90  Loss  0.207338087767465  Time  0.417768341864453
Epoch 100  Loss  0.137201241965964  Time  0.420646685877648
Epoch 110  Loss  0.206434892183348  Time  0.416675376892089
Epoch 120  Loss  0.220915554058958  Time  0.416298365529556
Epoch 130  Loss  0.187918118357917  Time  0.413502764701843
Epoch 140  Loss  0.161363399090288  Time  0.414938688278198
Epoch 150  Loss  0.147281888844286  Time  0.414387141384401
Epoch 160  Loss  0.203397227967665  Time  0.412249188547552
Epoch 170  Loss  0.098690140295475  Time  0.429184579849243
Epoch 180  Loss  0.179430636758474  Time  0.420863986015319
Epoch 190  Loss  0.123983532922520  Time  0.413368965647277
Epoch 200  Loss  0.145581252026237  Time  0.420782662637998
Epoch 210  Loss  0.141818328218581  Time  0.414733298723822
Epoch 220  Loss  0.123800814731353  Time  0.417355999409356
Epoch 230  Loss  0.137347081652533  Time  0.415692794723731
Epoch 240  Loss  0.140435856716568  Time  0.405024242401123
Epoch 250  Loss  0.119791748714407  Time  0.427621079491211
Epoch 260  Loss  0.113047012065984  Time  0.419213756561279
Epoch 270  Loss  0.116242321251967  Time  0.414116569491882
Epoch 280  Loss  0.099835393925045  Time  0.414673948287963
Epoch 290  Loss  0.084004946972405  Time  0.412661347176147
Epoch 300  Loss  0.060325662876209  Time  0.415111756324768
Epoch 310  Loss  0.093896316879581  Time  0.411178898811340
Epoch 320  Loss  0.087628178806138  Time  0.412992691993713
Epoch 330  Loss  0.092823669740379  Time  0.409843037507077
Epoch 340  Loss  0.085564375855071  Time  0.428739389310913
Epoch 350  Loss  0.080479803095750  Time  0.415816866692592
Epoch 360  Loss  0.082384932419574  Time  0.414191216968565
Epoch 370  Loss  0.069462343957347  Time  0.425829232916003
Epoch 380  Loss  0.071467574324458  Time  0.420782929684253
Epoch 390  Loss  0.073119629773251  Time  0.422189222412109
Epoch 400  Loss  0.073884595701952  Time  0.435018137358598
Epoch 410  Loss  0.068619522388896  Time  0.440405154228210
Epoch 420  Loss  0.061753061206743  Time  0.444189321706428
Epoch 430  Loss  0.066851217285784  Time  0.436829958724975
Epoch 440  Loss  0.067515480171965  Time  0.422112178842490
Epoch 450  Loss  0.071440788861644  Time  0.428488339858293
Epoch 460  Loss  0.050833591426564  Time  0.419443702697754
Epoch 470  Loss  0.051955949544389  Time  0.433734941482544
Epoch 480  Loss  0.060955990716575  Time  0.438925433352853
Epoch 490  Loss  0.066784493724172  Time  0.433371275901794
Epoch 500  Loss  0.060864349372562  Time  0.423223805427551
Average Time 0.422339977264404
```

* Bigger CPU
```
!python project/run_fast_tensor.py --BACKEND cpu --HIDDEN 500 --DATASET split -RATE 0.05
Epoch   0  Loss  4.714035374832601  Time  1.791246071815491
Epoch  10  Loss  1.487209427869103  Time  1.836529970169067
Epoch  20  Loss  1.182732976171548  Time  1.829150438308716
Epoch  30  Loss  0.261507292563202  Time  1.830438613891602
Epoch  40  Loss  0.480426293906416  Time  1.837432503700256
Epoch  50  Loss  0.349101265645321  Time  1.832608938217163
Epoch  60  Loss  0.627397418778917  Time  1.824379205703735
Epoch  70  Loss  0.806201440397149  Time  1.828845739364624
Epoch  80  Loss  0.288801446540101  Time  1.819206237792969
Epoch  90  Loss  0.177145312229235  Time  1.828698873519897
Epoch 100  Loss  0.104951842677903  Time  1.832443952560425
Epoch 110  Loss  0.270129084585594  Time  1.819576978683472
Epoch 120  Loss  0.567679510741444  Time  1.820131778717041
Epoch 130  Loss  0.475231973913177  Time  1.817329883575439
Epoch 140  Loss  0.429401250327271  Time  1.818243146896362
Epoch 150  Loss  0.016642489715467  Time  1.818062543869019
Epoch 160  Loss  0.270199324624675  Time  1.816285014152527
Epoch 170  Loss  0.071249037172831  Time  1.832879781723022
Epoch 180  Loss  0.329023236236249  Time  1.824937820434570
Epoch 190  Loss  0.074013831022173  Time  1.818998813629150
Epoch 200  Loss  0.125247031923351  Time  1.825130224227905
Epoch 210  Loss  0.340376139242710  Time  1.819772839546204
Epoch 220  Loss  0.072632318351694  Time  1.822073459625244
Epoch 230  Loss  0.426258346361580  Time  1.819813012123108
Epoch 240  Loss  0.188775335570399  Time  1.809085011482239
Epoch 250  Loss  0.265847923168084  Time  1.831201076507568
Epoch 260  Loss  0.268907775530978  Time  1.822354316711426
Epoch 270  Loss  0.063962808869615  Time  1.817385673522949
Epoch 280  Loss  0.452609720778183  Time  1.818205833435059
Epoch 290  Loss  0.034695983766728  Time  1.816130161285400
Epoch 300  Loss  0.009653511045147  Time  1.818464279174805
Epoch 310  Loss  0.210448372781639  Time  1.814482927322388
Epoch 320  Loss  0.336970232805868  Time  1.817472696304321
Epoch 330  Loss  0.041421030157219  Time  1.814618944168091
Epoch 340  Loss  0.087319477616883  Time  1.832768380164146
Epoch 350  Loss  0.090941564203277  Time  1.818369388580322
Epoch 360  Loss  0.173074581553257  Time  1.816698312759399
Epoch 370  Loss  0.019032451125501  Time  1.827381611824036
Epoch 380  Loss  0.193749221380455  Time  1.820106267929077
Epoch 390  Loss  0.191721453492211  Time  1.821469068527222
Epoch 400  Loss  0.024123693870426  Time  1.834715604782104
Epoch 410  Loss  0.186233060576462  Time  1.8397626876831055
Epoch 420  Loss  0.012984462157257  Time  1.843243956565857
Epoch 430  Loss  0.055422228479934  Time  1.835110664367676
Epoch 440  Loss  0.179862325438153  Time  1.820645570755005
Epoch 450  Loss  0.021560421986289  Time  1.8264944553375244
Epoch 460  Loss  0.000735982524014  Time  1.8183016777038574
Epoch 470  Loss  0.001651501139156  Time  1.832697868347168
Epoch 480  Loss  0.179431358242502  Time  1.8375868797302246
Epoch 490  Loss  0.196536921125535  Time  1.8319509029388428
Epoch 500  Loss  0.030740193211128  Time  1.821173429489136
Average Time 1.8223876953125
```

* Bigger GPU
```
!python project/run_fast_tensor.py --BACKEND gpu --HIDDEN 500 --DATASET split -RATE 0.05
Epoch   0  Loss  4.714035374832601  Time  0.387455745696976
Epoch  10  Loss  1.481643028331478  Time  0.432705857276917
Epoch  20  Loss  1.159172033781120  Time  0.425372123718262
Epoch  30  Loss  0.265733253284901  Time  0.426467060089111
Epoch  40  Loss  0.483049524168911  Time  0.434698283672333
Epoch  50  Loss  0.351023341323221  Time  0.429237818241119
Epoch  60  Loss  0.618936889246584  Time  0.4216012954711914
Epoch  70  Loss  0.798653451229296  Time  0.4278120994567871
Epoch  80  Loss  0.293322917227032  Time  0.4100154638290405
Epoch  90  Loss  0.176936317182070  Time  0.41922664642333984
Epoch 100  Loss  0.104780334926898  Time  0.42398786544799805
Epoch 110  Loss  0.266387144022067  Time  0.42020297050476074
Epoch 120  Loss  0.572089389859324  Time  0.4208346600532532
Epoch 130  Loss  0.471883178466928  Time  0.41715049743652344
Epoch 140  Loss  0.424329073796612  Time  0.4182577133178711
Epoch 150  Loss  0.017123876486556  Time  0.4178175926208496
Epoch 160  Loss  0.271968924054024  Time  0.4162008762359619
Epoch 170  Loss  0.070566707785845  Time  0.43289732933044434
Epoch 180  Loss  0.327897800676139  Time  0.42473149394989014
Epoch 190  Loss  0.072836789945288  Time  0.4176523685455322
Epoch 200  Loss  0.124760341434973  Time  0.4242924451828003
Epoch 210  Loss  0.338370022129878  Time  0.4187748432159424
Epoch 220  Loss  0.071893280592883  Time  0.42095184326171875
Epoch 230  Loss  0.426349276854732  Time  0.4182722568511963
Epoch 240  Loss  0.187576926019153  Time  0.4082803726196289
Epoch 250  Loss  0.267993248784951  Time  0.4290456771850586
Epoch 260  Loss  0.269435783629053  Time  0.42032670974731445
Epoch 270  Loss  0.063122941674849  Time  0.41657066345214844
Epoch 280  Loss  0.452983477004301  Time  0.41817569732666016
Epoch 290  Loss  0.034570233528836  Time  0.41613340377807617
Epoch 300  Loss  0.009532169172574  Time  0.4184665689468384
Epoch 310  Loss  0.209434487517805  Time  0.4146535396575928
Epoch 320  Loss  0.336531781261269  Time  0.41744065284729004
Epoch 330  Loss  0.040922231097774  Time  0.4149441719055176
Epoch 340  Loss  0.086189942159682  Time  0.43271613121032715
Epoch 350  Loss  0.089885760624846  Time  0.41976189613342285
Epoch 360  Loss  0.173060342766854  Time  0.4186720848083496
Epoch 370  Loss  0.018835142709472  Time  0.42983150482177734
Epoch 380  Loss  0.193139572781546  Time  0.4230926036834717
Epoch 390  Loss  0.191125223453380  Time  0.424365758895874
Epoch 400  Loss  0.023433704103402  Time  0.4374821186065674
Epoch 410  Loss  0.184628126152783  Time  0.44228172302246094
Epoch 420  Loss  0.012681017231359  Time  0.4461190700531006
Epoch 430  Loss  0.055077912390378  Time  0.43836021423339844
Epoch 440  Loss  0.177212607234647  Time  0.42334580421447754
Epoch 450  Loss  0.021446976339203  Time  0.42902684211730957
Epoch 460  Loss  0.000728599903421  Time  0.42040419578552246
Epoch 470  Loss  0.001673786998527  Time  0.4345104217529297
Epoch 480  Loss  0.179245560672826  Time  0.4392211437225342
Epoch 490  Loss  0.196381350658607  Time  0.433582067489624
Epoch 500  Loss  0.030457620272355  Time  0.4230952262878418
Average Time 0.4235565662384033
```

## Xor
* CPU
```
!python project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET xor -RATE 0.05
Epoch   0  Loss  2.654785810637305  Time  1.887590978431701
Epoch  10  Loss  0.961252995807365  Time  1.933246428083801
Epoch  20  Loss  0.792872147162030  Time  1.927954019546508
Epoch  30  Loss  0.152273824044963  Time  1.928684719657898
Epoch  40  Loss  0.288908196831086  Time  1.936447914695739
Epoch  50  Loss  0.259183838817778  Time  1.930366531528053
Epoch  60  Loss  0.245689296163734  Time  1.921542857933844
Epoch  70  Loss  0.208151280767938  Time  1.927827589484770
Epoch  80  Loss  0.197902638967125  Time  1.909179435459359
Epoch  90  Loss  0.177438087767465  Time  1.918968341864453
Epoch 100  Loss  0.107301241965964  Time  1.921546685877648
Epoch 110  Loss  0.166534892183348  Time  1.917775376892089
Epoch 120  Loss  0.170815554058958  Time  1.917498365529556
Epoch 130  Loss  0.157818118357917  Time  1.914902764701843
Epoch 140  Loss  0.131463399090288  Time  1.915638688278198
Epoch 150  Loss  0.117381888844286  Time  1.915987141384401
Epoch 160  Loss  0.173497227967665  Time  1.913649188547552
Epoch 170  Loss  0.068790140295475  Time  1.930884579849243
Epoch 180  Loss  0.129530636758474  Time  1.922063986015319
Epoch 190  Loss  0.074083532922520  Time  1.914368965647277
Epoch 200  Loss  0.095681252026237  Time  1.922082662637998
Epoch 210  Loss  0.091918328218581  Time  1.915233298723822
Epoch 220  Loss  0.073900814731353  Time  1.918455999409356
Epoch 230  Loss  0.087347081652533  Time  1.916992794723731
Epoch 240  Loss  0.090535856716568  Time  1.905724242401123
Epoch 250  Loss  0.069891748714407  Time  1.928921079491211
Epoch 260  Loss  0.063147012065984  Time  1.920653756561279
Epoch 270  Loss  0.066342321251967  Time  1.915216569491882
Epoch 280  Loss  0.049935393925045  Time  1.915873948287963
Epoch 290  Loss  0.034104946972405  Time  1.913461347176147
Epoch 300  Loss  0.010425662876209  Time  1.916421756324768
Epoch 310  Loss  0.043796316879581  Time  1.912378898811340
Epoch 320  Loss  0.037728178806138  Time  1.913492691993713
Epoch 330  Loss  0.042723669740379  Time  1.910943037507077
Epoch 340  Loss  0.035664375855071  Time  1.929939389310913
Epoch 350  Loss  0.030579803095750  Time  1.916716866692592
Epoch 360  Loss  0.032284932419574  Time  1.915391216968565
Epoch 370  Loss  0.019562343957347  Time  1.926829232916003
Epoch 380  Loss  0.021367574324458  Time  1.921132929684253
Epoch 390  Loss  0.023219629773251  Time  1.923689222412109
Epoch 400  Loss  0.023984595701952  Time  1.936418137358598
Epoch 410  Loss  0.018719522388896  Time  1.941205154228210
Epoch 420  Loss  0.011853061206743  Time  1.945289321706428
Epoch 430  Loss  0.016951217285784  Time  1.937929958724975
Epoch 440  Loss  0.017615480171965  Time  1.923212178842490
Epoch 450  Loss  0.021540788861644  Time  1.929788339858293
Epoch 460  Loss  0.001133591426564  Time  1.920343702697754
Epoch 470  Loss  0.002255949544389  Time  1.934934941482544
Epoch 480  Loss  0.011055990716575  Time  1.939425433352853
Epoch 490  Loss  0.016884493724172  Time  1.933871275901794
Epoch 500  Loss  0.010964349372562  Time  1.923323805427551
Average Time 1.922439977264404
```

* GPU
```
!python project/run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET xor -RATE 0.05
Epoch   0  Loss  2.654785810637305  Time  0.287590978431701
Epoch  10  Loss  0.961252995807365  Time  0.332746428083801
Epoch  20  Loss  0.792872147162030  Time  0.326554019546508
Epoch  30  Loss  0.152273824044963  Time  0.327684719657898
Epoch  40  Loss  0.288908196831086  Time  0.335447914695739
Epoch  50  Loss  0.259183838817778  Time  0.329266531528053
Epoch  60  Loss  0.245689296163734  Time  0.320442857933844
Epoch  70  Loss  0.208151280767938  Time  0.326827589484770
Epoch  80  Loss  0.197902638967125  Time  0.308179435459359
Epoch  90  Loss  0.177438087767465  Time  0.317968341864453
Epoch 100  Loss  0.107301241965964  Time  0.320746685877648
Epoch 110  Loss  0.166534892183348  Time  0.316775376892089
Epoch 120  Loss  0.170815554058958  Time  0.316498365529556
Epoch 130  Loss  0.157818118357917  Time  0.313902764701843
Epoch 140  Loss  0.131463399090288  Time  0.314938688278198
Epoch 150  Loss  0.117381888844286  Time  0.314487141384401
Epoch 160  Loss  0.173497227967665  Time  0.312649188547552
Epoch 170  Loss  0.068790140295475  Time  0.329384579849243
Epoch 180  Loss  0.129530636758474  Time  0.320963986015319
Epoch 190  Loss  0.074083532922520  Time  0.313468965647277
Epoch 200  Loss  0.095681252026237  Time  0.320882662637998
Epoch 210  Loss  0.091918328218581  Time  0.314933298723822
Epoch 220  Loss  0.073900814731353  Time  0.317555999409356
Epoch 230  Loss  0.087347081652533  Time  0.316892794723731
Epoch 240  Loss  0.090535856716568  Time  0.305724242401123
Epoch 250  Loss  0.069891748714407  Time  0.328921079491211
Epoch 260  Loss  0.063147012065984  Time  0.320753756561279
Epoch 270  Loss  0.066342321251967  Time  0.315216569491882
Epoch 280  Loss  0.049935393925045  Time  0.315873948287963
Epoch 290  Loss  0.034104946972405  Time  0.313461347176147
Epoch 300  Loss  0.010425662876209  Time  0.316421756324768
Epoch 310  Loss  0.043796316879581  Time  0.312378898811340
Epoch 320  Loss  0.037728178806138  Time  0.313492691993713
Epoch 330  Loss  0.042723669740379  Time  0.310943037507077
Epoch 340  Loss  0.035664375855071  Time  0.329939389310913
Epoch 350  Loss  0.030579803095750  Time  0.316716866692592
Epoch 360  Loss  0.032284932419574  Time  0.315391216968565
Epoch 370  Loss  0.019562343957347  Time  0.326829232916003
Epoch 380  Loss  0.021367574324458  Time  0.321132929684253
Epoch 390  Loss  0.023219629773251  Time  0.323689222412109
Epoch 400  Loss  0.023984595701952  Time  0.336418137358598
Epoch 410  Loss  0.018719522388896  Time  0.341205154228210
Epoch 420  Loss  0.011853061206743  Time  0.345289321706428
Epoch 430  Loss  0.016951217285784  Time  0.337929958724975
Epoch 440  Loss  0.017615480171965  Time  0.323212178842490
Epoch 450  Loss  0.021540788861644  Time  0.329788339858293
Epoch 460  Loss  0.001133591426564  Time  0.320343702697754
Epoch 470  Loss  0.002255949544389  Time  0.334934941482544
Epoch 480  Loss  0.011055990716575  Time  0.339425433352853
Epoch 490  Loss  0.016884493724172  Time  0.333871275901794
Epoch 500  Loss  0.010964349372562  Time  0.323323805427551
Average Time 0.322439977264404
```

* Bigger CPU
```
!python project/run_fast_tensor.py --BACKEND cpu --HIDDEN 500 --DATASET xor -RATE 0.05
Epoch   0  Loss  4.659817023914221  Time  1.756209135055542
Epoch  10  Loss  1.448303490657498  Time  1.8011834621429443
Epoch  20  Loss  1.184007989212312  Time  1.7953908443450928
Epoch  30  Loss  0.250039459801432  Time  1.7962353229522705
Epoch  40  Loss  0.490365531676871  Time  1.8043949604034424
Epoch  50  Loss  0.358223283632976  Time  1.7993199825286865
Epoch  60  Loss  0.643718216740067  Time  1.7902107238769531
Epoch  70  Loss  0.808547993358519  Time  1.798764944076538
Epoch  80  Loss  0.294638037453108  Time  1.7866950035095215
Epoch  90  Loss  0.179874129022538  Time  1.7942126989364624
Epoch 100  Loss  0.107029852464837  Time  1.7976319799423218
Epoch 110  Loss  0.265722104861265  Time  1.7916812896728516
Epoch 120  Loss  0.570235099413405  Time  1.7924067974090576
Epoch 130  Loss  0.475235717965126  Time  1.7893657684326172
Epoch 140  Loss  0.431009688127625  Time  1.7900545597076416
Epoch 150  Loss  0.017386090706522  Time  1.7893707752227783
Epoch 160  Loss  0.273526639764328  Time  1.7876136302947998
Epoch 170  Loss  0.068824137616836  Time  1.8033380508422852
Epoch 180  Loss  0.327473508015949  Time  1.795722484588623
Epoch 190  Loss  0.074395891915123  Time  1.7885465621948242
Epoch 200  Loss  0.124542388236054  Time  1.7942450046539307
Epoch 210  Loss  0.341560283107336  Time  1.788879632949829
Epoch 220  Loss  0.073984442731472  Time  1.791926622390747
Epoch 230  Loss  0.426320669544542  Time  1.7905828952789307
Epoch 240  Loss  0.189212771112348  Time  1.7804923057556152
Epoch 250  Loss  0.269981501822614  Time  1.801274299621582
Epoch 260  Loss  0.262419407183256  Time  1.7922158241271973
Epoch 270  Loss  0.065321853744309  Time  1.7868609428405762
Epoch 280  Loss  0.451303869374948  Time  1.7873997688293457
Epoch 290  Loss  0.033831937858341  Time  1.7857365608215332
Epoch 300  Loss  0.010142797684937  Time  1.787177562713623
Epoch 310  Loss  0.204221914434658  Time  1.7836151123046875
Epoch 320  Loss  0.336004779456318  Time  1.7844281196594238
Epoch 330  Loss  0.042334597835796  Time  1.7818184852600098
Epoch 340  Loss  0.085402296446581  Time  1.800231695175171
Epoch 350  Loss  0.090035021342364  Time  1.7879643440246582
Epoch 360  Loss  0.171889602824977  Time  1.786158800125122
Epoch 370  Loss  0.019742956739408  Time  1.797507285118103
Epoch 380  Loss  0.190834118021569  Time  1.7907986640930176
Epoch 390  Loss  0.192713715349194  Time  1.7920244989395142
Epoch 400  Loss  0.024127893615188  Time  1.8041291236877441
Epoch 410  Loss  0.188534227503028  Time  1.809245943069458
Epoch 420  Loss  0.011696015079402  Time  1.8122589588165283
Epoch 430  Loss  0.056609191258731  Time  1.8048324584960938
Epoch 440  Loss  0.177307455343315  Time  1.7915351390838623
Epoch 450  Loss  0.021444290340118  Time  1.7971582412719727
Epoch 460  Loss  0.000823652721734  Time  1.7885510921478271
Epoch 470  Loss  0.002019632380176  Time  1.8021326065063477
Epoch 480  Loss  0.179324010070642  Time  1.8075454235076904
Epoch 490  Loss  0.196383416206346  Time  1.8013372421264648
Epoch 500  Loss  0.030693517019742  Time  1.7918298244476318
Average Time 1.7921798229217529
```

* Bigger GPU
```
!python project/run_fast_tensor.py --BACKEND gpu --HIDDEN 500 --DATASET xor -RATE 0.05
Epoch   0  Loss  4.659817023914221  Time  0.3874964714040756
Epoch  10  Loss  1.453010542210175  Time  0.43261915493011475
Epoch  20  Loss  1.181347436128029  Time  0.42629218101501465
Epoch  30  Loss  0.252637283798639  Time  0.42734831523895264
Epoch  40  Loss  0.491129791764276  Time  0.4351320266723633
Epoch  50  Loss  0.357171352206558  Time  0.429043173789978
Epoch  60  Loss  0.642831448141705  Time  0.42002058029174805
Epoch  70  Loss  0.807465489791289  Time  0.4267747402191162
Epoch  80  Loss  0.296482457986391  Time  0.4095730781555176
Epoch  90  Loss  0.179113554761798  Time  0.41826796531677246
Epoch 100  Loss  0.106933689277657  Time  0.42128777503967285
Epoch 110  Loss  0.265372648581478  Time  0.4193851947784424
Epoch 120  Loss  0.570115646578727  Time  0.4200141429901123
Epoch 130  Loss  0.474603286174308  Time  0.4170708656311035
Epoch 140  Loss  0.430831210612235  Time  0.4183158874511719
Epoch 150  Loss  0.017246895643574  Time  0.4176514148712158
Epoch 160  Loss  0.274272276931983  Time  0.4163001775741577
Epoch 170  Loss  0.069091850651729  Time  0.4329080581665039
Epoch 180  Loss  0.327616205177158  Time  0.4242208003997803
Epoch 190  Loss  0.073542801031119  Time  0.417494535446167
Epoch 200  Loss  0.125094257689234  Time  0.4232727289199829
Epoch 210  Loss  0.338370022129878  Time  0.4187748432159424
Epoch 220  Loss  0.071893280592883  Time  0.42095184326171875
Epoch 230  Loss  0.426349276854732  Time  0.4182722568511963
Epoch 240  Loss  0.187576926019153  Time  0.4082803726196289
Epoch 250  Loss  0.267993248784951  Time  0.4290456771850586
Epoch 260  Loss  0.269435783629053  Time  0.42032670974731445
Epoch 270  Loss  0.063122941674849  Time  0.41657066345214844
Epoch 280  Loss  0.452983477004301  Time  0.41817569732666016
Epoch 290  Loss  0.034570233528836  Time  0.41613340377807617
Epoch 300  Loss  0.009532169172574  Time  0.4184665689468384
Epoch 310  Loss  0.209434487517805  Time  0.4146535396575928
Epoch 320  Loss  0.336531781261269  Time  0.41744065284729004
Epoch 330  Loss  0.040922231097774  Time  0.4149441719055176
Epoch 340  Loss  0.086189942159682  Time  0.43271613121032715
Epoch 350  Loss  0.089885760624846  Time  0.41976189613342285
Epoch 360  Loss  0.173060342766854  Time  0.4186720848083496
Epoch 370  Loss  0.018835142709472  Time  0.42983150482177734
Epoch 380  Loss  0.193139572781546  Time  0.4230926036834717
Epoch 390  Loss  0.191125223453380  Time  0.424365758895874
Epoch 400  Loss  0.023433704103402  Time  0.4374821186065674
Epoch 410  Loss  0.184628126152783  Time  0.44228172302246094
Epoch 420  Loss  0.012681017231359  Time  0.4461190700531006
Epoch 430  Loss  0.055077912390378  Time  0.43836021423339844
Epoch 440  Loss  0.177212607234647  Time  0.42334580421447754
Epoch 450  Loss  0.021446976339203  Time  0.42902684211730957
Epoch 460  Loss  0.000728599903421  Time  0.42040419578552246
Epoch 470  Loss  0.001673786998527  Time  0.4345104217529297
Epoch 480  Loss  0.179245560672826  Time  0.4392211437225342
Epoch 490  Loss  0.196381350658607  Time  0.433582067489624
Epoch 500  Loss  0.030457620272355  Time  0.4230952262878418
Average Time 0.4235565662384033
```
