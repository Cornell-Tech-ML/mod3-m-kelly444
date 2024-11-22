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
```

## Xor
* CPU
```
```

* GPU
```
```

* Bigger CPU
```
```

* Bigger GPU
```
```
