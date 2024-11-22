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

* GPU
```
Epoch   0 loss  0.108842962768063  correct 42 time taken 1.385655185604858
Epoch  10 loss  2.519090535386004  correct 46 time taken 1.888670960985995
Epoch  20 loss  1.321409733657032  correct 48 time taken 1.879656767845154
Epoch  30 loss  0.913366939896401  correct 48 time taken 1.853362822534537
Epoch  40 loss  0.620092825246209  correct 48 time taken 1.856319022178649
Epoch  50 loss  0.413130167703824  correct 49 time taken 1.859401655517435
Epoch  60 loss  0.164671742001779  correct 49 time taken 1.851230072975158
Epoch  70 loss  1.260069129787749  correct 48 time taken 1.841674876233744
Epoch  80 loss  0.135528424028269  correct 48 time taken 1.839575982093811
Epoch  90 loss  0.967130488679463  correct 49 time taken 1.845329666137650
Epoch 100 loss  0.554510010831735  correct 49 time taken 1.841510474192557
Epoch 110 loss  1.094919032534728  correct 50 time taken 1.848481838962936
Epoch 120 loss  0.177937794129636  correct 48 time taken 1.850868010528033
Epoch 130 loss  0.045842846879523  correct 49 time taken 1.843524956783186
Epoch 140 loss  0.103098270192577  correct 49 time taken 1.839810943519157
Epoch 150 loss  0.642282116365560  correct 49 time taken 1.847510552486311
Epoch 160 loss  0.827702820488992  correct 49 time taken 1.846319270133795
Epoch 170 loss  0.608920759881437  correct 48 time taken 1.848809623718261
Epoch 180 loss  0.420353615780084  correct 48 time taken 1.851344561578946
Epoch 190 loss  0.803247515148195  correct 48 time taken 1.856842374801635
Epoch 200 loss  0.519655359826164  correct 48 time taken 1.845846033809313
Epoch 210 loss  0.167362363964736  correct 49 time taken 1.847084975242614
Epoch 220 loss  0.563793804423735  correct 48 time taken 1.850472107879264
Epoch 230 loss  0.956557732533613  correct 50 time taken 1.847050870762634
Epoch 240 loss  0.263659474043851  correct 49 time taken 1.842265186412454
Epoch 250 loss  0.280851919059925  correct 50 time taken 1.851391823523525
Epoch 260 loss  0.384382955397294  correct 48 time taken 1.845586362373809
Epoch 270 loss  0.325486143201183  correct 49 time taken 1.844377069478985
Epoch 280 loss  0.346111885211834  correct 48 time taken 1.843424153327941
Epoch 290 loss  0.048510189820519  correct 48 time taken 1.845042061589725
Epoch 300 loss  0.066379391386009  correct 48 time taken 1.844865846633911
Epoch 310 loss  0.821631065924063  correct 49 time taken 1.848065288943806
Epoch 320 loss  0.054769528948927  correct 49 time taken 1.857471895217895
Epoch 330 loss  0.106850756816844  correct 49 time taken 1.845507907842431
Epoch 340 loss  0.389438829884605  correct 48 time taken 1.859937429428100
Epoch 350 loss  0.821101468756326  correct 49 time taken 1.842382216856527
Epoch 360 loss  0.046448002645683  correct 48 time taken 1.848680400848388
Epoch 370 loss  0.027621112500523  correct 49 time taken 1.842765287658691
Epoch 380 loss  1.045225912284845  correct 49 time taken 1.845326553068034
Epoch 390 loss  0.553997148387415  correct 49 time taken 1.842995988045349
Epoch 400 loss  0.014443680891575  correct 49 time taken 1.849735527577025
Epoch 410 loss  0.867392094730862  correct 49 time taken 1.848480510711167
Epoch 420 loss  0.909816718234253  correct 50 time taken 1.848941135469494
Epoch 430 loss  0.292782634288337  correct 49 time taken 1.843468546867370
Epoch 440 loss  0.423097456535674  correct 48 time taken 1.841446325050848
Epoch 450 loss  0.067455477455623  correct 49 time taken 1.841761541366577
Epoch 460 loss  0.835043107130182  correct 48 time taken 1.845311331748962
Epoch 470 loss  0.001605016523869  correct 49 time taken 1.844616365432739
Epoch 480 loss  0.919738357545767  correct 49 time taken 1.853280639643664
Epoch 490 loss  0.025420700216215  correct 49 time taken 1.854887032508085
Epoch 500 loss  0.835355004078906  correct 49 time taken 1.842503328917358
Average Time Taken  1.856419139385223
```

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
* Bigger GPU

## Spiral
* CPU
* GPU
* Bigger CPU
* Bigger GPU


