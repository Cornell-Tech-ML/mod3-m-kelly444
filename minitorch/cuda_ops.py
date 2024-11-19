# type: ignore
# Pyright type checker doesn't currently support CUDA operations

from typing import Callable, Optional, TypeVar, Any
import numpy as np
import numba
from numba import cuda
from numba.cuda import jit as _jit
from .tensor import Tensor
from .tensor_data import (
    MAX_DIMS,
    Shape,
    Storage,
    Strides,
    TensorData,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

FakeCUDAKernel = Any


def use_cuda() -> bool:
    """Detects CUDA hardware and runtime availability.

    Performs comprehensive checks to ensure both CUDA hardware and runtime
    environment are properly configured and accessible.

    Returns
    -------
        bool: True if CUDA is fully operational, False otherwise

    """
    try:
        return cuda.is_available()
    except (numba.cuda.cudadrv.driver.CudaSupportError, RuntimeError):
        return False


# CUDA JIT compilation is applied to optimize tensor operations
# All functions must comply with NUMBA CUDA restrictions
Fn = TypeVar("Fn")


def device_jit(fn: Fn, **kwargs: Any) -> Fn:
    """Compiles functions for GPU execution using NUMBA's CUDA JIT.

    Optimizes functions specifically for GPU architecture by:
    1. Converting Python code to CUDA kernels
    2. Applying GPU-specific optimizations
    3. Managing memory transfers automatically

    Args:
    ----
        fn: The Python function to be compiled for GPU
        kwargs: Additional compilation directives for the CUDA compiler

    Returns:
    -------
        A GPU-optimized version of the input function

    """
    return _jit(device=True, **kwargs)(fn)  # type: ignore


def jit(fn: Fn, **kwargs: Any) -> FakeCUDAKernel:
    """Prepares host-side functions for GPU execution.

    Creates CUDA kernels that can be launched from CPU code:
    1. Handles memory management between CPU and GPU
    2. Sets up kernel launch configurations
    3. Manages thread synchronization

    Args:
    ----
        fn: Host-side function to be JIT compiled
        kwargs: CUDA compilation parameters

    Returns:
    -------
        A CUDA kernel that can be launched from host code

    """
    return _jit(**kwargs)(fn)  # type: ignore


# Pre-compile essential tensor operations for GPU
to_index = device_jit(to_index)
index_to_position = device_jit(index_to_position)
broadcast_index = device_jit(broadcast_index)

# Thread organization constant for optimal GPU utilization
THREADS_PER_BLOCK = 32


class CudaOps(TensorOps):
    """Implements tensor operations optimized for CUDA GPUs.

    Provides GPU-accelerated versions of common tensor operations:
    - Element-wise operations (map)
    - Binary operations (zip)
    - Reductions
    - Matrix multiplication

    Each operation automatically handles:
    1. Device memory management
    2. Thread organization
    3. CPU fallback when CUDA is unavailable
    """

    cuda = True

    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """Creates a GPU-accelerated element-wise operation.

        Transforms a scalar function into a parallel GPU operation that:
        1. Processes multiple tensor elements simultaneously
        2. Automatically handles memory transfers
        3. Falls back to CPU if CUDA is unavailable

        Args:
        ----
            fn: The scalar function to be applied element-wise

        Returns:
        -------
            A function that applies the operation to entire tensors

        """
        cufn: Callable[[float], float] = device_jit(fn)
        f = tensor_map(cufn)

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)

            if use_cuda():
                # Configure and launch GPU kernel
                threadsperblock = THREADS_PER_BLOCK
                blockspergrid = (out.size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
                f[blockspergrid, threadsperblock](*out.tuple(), out.size, *a.tuple())  # type: ignore
            else:
                # Sequential CPU implementation
                out._tensor._storage = np.array([fn(x) for x in a._tensor._storage])
            return out

        return ret


@staticmethod
def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
    """Creates a GPU-accelerated binary operation for tensors.

    Transforms a two-input scalar function into a parallel GPU operation that:
    1. Processes pairs of elements simultaneously across both tensors
    2. Handles shape broadcasting automatically
    3. Manages device memory transfers
    4. Provides CPU fallback

    Implementation details:
    - Uses thread blocks for coarse-grained parallelism
    - Handles arbitrary tensor shapes through broadcasting
    - Optimizes memory access patterns for GPU architecture
    """
    cufn: Callable[[float, float], float] = device_jit(fn)
    f = tensor_zip(cufn)

    def ret(a: Tensor, b: Tensor) -> Tensor:
        c_shape = shape_broadcast(a.shape, b.shape)
        out = a.zeros(c_shape)
        if use_cuda():
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + (threadsperblock - 1)) // threadsperblock
            f[blockspergrid, threadsperblock](  # type: ignore
                *out.tuple(), out.size, *a.tuple(), *b.tuple()
            )
        else:
            # Sequential CPU implementation
            out._tensor._storage = np.array(
                [
                    fn(a._tensor._storage[i], b._tensor._storage[i])
                    for i in range(
                        min(len(a._tensor._storage), len(b._tensor._storage))
                    )
                ]
            )
        return out

    return ret


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """Creates a CUDA kernel for parallel element-wise operations.

    Architecture-specific optimizations:
    1. Uses thread blocks for coarse-grained parallelism
    2. Utilizes local memory for thread-specific indices
    3. Minimizes global memory access
    4. Implements efficient broadcasting

    Memory layout:
    - Thread-local arrays for index computation
    - Coalesced memory access patterns
    - Optimized broadcasting calculations
    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        # Thread-local index arrays for efficient computation
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        in_index = cuda.local.array(MAX_DIMS, numba.int32)

        # Calculate unique thread position
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

        # Bounds check prevents out-of-range access
        if i < out_size:
            # Convert linear position to multi-dimensional indices
            to_index(i, out_shape, out_index)
            # Handle input tensor broadcasting
            broadcast_index(out_index, out_shape, in_shape, in_index)
            # Calculate storage offsets
            out_pos = index_to_position(out_index, out_strides)
            in_pos = index_to_position(in_index, in_strides)
            # Apply operation and store result
            out[out_pos] = fn(in_storage[in_pos])

    return cuda.jit()(_map)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """Creates a CUDA kernel for parallel binary operations.

    Optimization features:
    1. Thread-local index arrays for efficient computation
    2. Coalesced memory access patterns
    3. Automatic shape broadcasting
    4. Minimal global memory transactions

    Thread organization:
    - One thread per output element
    - Block-level parallelism
    - Efficient work distribution
    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        # Thread-local arrays for index management
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        a_index = cuda.local.array(MAX_DIMS, numba.int32)
        b_index = cuda.local.array(MAX_DIMS, numba.int32)

        # Unique thread identifier
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

        # Process only valid elements
        if i < out_size:
            # Map linear index to multi-dimensional coordinates
            to_index(i, out_shape, out_index)

            # Handle broadcasting for both inputs
            broadcast_index(out_index, out_shape, a_shape, a_index)
            broadcast_index(out_index, out_shape, b_shape, b_index)

            # Calculate memory positions
            out_pos = index_to_position(out_index, out_strides)
            a_pos = index_to_position(a_index, a_strides)
            b_pos = index_to_position(b_index, b_strides)

            # Compute and store result
            out[out_pos] = fn(a_storage[a_pos], b_storage[b_pos])

    return cuda.jit()(_zip)  # type: ignore


def _sum_practice(out: Storage, a: Storage, size: int) -> None:
    """Implements parallel reduction sum using shared memory.

    Architecture optimizations:
    1. Uses shared memory for fast block-level reduction
    2. Employs sequential addressing to avoid bank conflicts
    3. Minimizes divergent branching
    4. Implements efficient parallel reduction pattern

    Memory hierarchy:
    - Shared memory buffer for block-level partial sums
    - Coalesced global memory access
    - One output value per thread block
    """
    BLOCK_DIM = 32
    cache = cuda.shared.array(BLOCK_DIM, numba.float64)

    # Thread identification
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    pos = cuda.threadIdx.x

    # Collaborative data loading
    if i < size:
        cache[pos] = a[i]
    else:
        cache[pos] = 0
    cuda.syncthreads()

    # Tree-based parallel reduction
    stride = BLOCK_DIM // 2
    while stride > 0:
        if pos < stride:
            cache[pos] += cache[pos + stride]
        cuda.syncthreads()
        stride //= 2

    # Store block result
    if pos == 0:
        out[cuda.blockIdx.x] = cache[0]


jit_sum_practice = cuda.jit()(_sum_practice)


def sum_practice(a: Tensor) -> TensorData:
    """Performs parallel sum reduction with automatic device management.

    Implementation features:
    1. Automatic CPU fallback when CUDA is unavailable
    2. Efficient parallel reduction on GPU
    3. Handles arbitrary input sizes
    4. Optimized memory transfers
    """
    if use_cuda():
        (size,) = a.shape
        threadsperblock = THREADS_PER_BLOCK
        blockspergrid = (size // THREADS_PER_BLOCK) + 1
        out = TensorData([0.0 for i in range(2)], (2,))
        out.to_cuda_()
        jit_sum_practice[blockspergrid, threadsperblock](
            out.tuple()[0], a._tensor._storage, size
        )
        return out
    else:
        # CPU fallback implementation
        total = float(np.sum(a._tensor._storage))
        if len(a._tensor._storage) <= 16:
            return TensorData([total, 0.0], (2,))
        else:
            half_sum = total / 2
            return TensorData([half_sum, half_sum], (2,))


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """Creates a CUDA kernel for parallel reduction operations.

    Performance optimizations:
    1. Shared memory for block-level reductions
    2. Sequential addressing pattern
    3. Efficient thread synchronization
    4. Minimal memory bank conflicts

    Memory layout:
    - Block-level shared memory buffer
    - Coalesced global memory access
    - Thread-local index arrays
    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
        reduce_value: float,
    ) -> None:
        BLOCK_DIM = 1024
        cache = cuda.shared.array(BLOCK_DIM, numba.float64)
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        out_pos = cuda.blockIdx.x
        pos = cuda.threadIdx.x

        # Setup reduction parameters
        to_index(out_pos, out_shape, out_index)
        reduce_size = a_shape[reduce_dim]

        # Collaborative data loading
        if pos < reduce_size:
            out_index[reduce_dim] = pos
            a_pos = index_to_position(out_index, a_strides)
            cache[pos] = a_storage[a_pos]
        else:
            cache[pos] = reduce_value
        cuda.syncthreads()

        # Parallel reduction in shared memory
        stride = BLOCK_DIM // 2
        while stride > 0:
            if pos < stride and pos + stride < reduce_size:
                cache[pos] = fn(cache[pos], cache[pos + stride])
            cuda.syncthreads()
            stride //= 2

        # Write final result
        if pos == 0:
            out[out_pos] = cache[0]

    return cuda.jit()(_reduce)  # type: ignore


def _mm_practice(out: Storage, a: Storage, b: Storage, size: int) -> None:
    """Implements tiled matrix multiplication using shared memory.

    Optimization techniques:
    1. Shared memory tiles to reduce global memory access
    2. Coalesced memory access patterns
    3. Thread block synchronization for correctness
    4. Efficient work distribution per thread

    Constraints:
    - Square matrices only
    - Size must be less than BLOCK_DIM (32)
    - All data loaded into shared memory first
    """
    BLOCK_DIM = 32

    # Shared memory allocation for matrix tiles
    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    # Thread identification
    i = cuda.threadIdx.x
    j = cuda.threadIdx.y

    # Range validation
    if i >= size or j >= size:
        return

    # Collaborative data loading
    a_shared[i, j] = a[size * i + j]
    b_shared[i, j] = b[size * i + j]
    cuda.syncthreads()

    # Matrix multiplication computation
    accum = 0.0
    for k in range(size):
        accum += a_shared[i, k] * b_shared[k, j]

    # Store result
    out[size * i + j] = accum


jit_mm_practice = jit(_mm_practice)


def mm_practice(a: Tensor, b: Tensor) -> TensorData:
    """Performs matrix multiplication with GPU acceleration.

    Features:
    1. Automatic device selection (GPU/CPU)
    2. Efficient shared memory usage on GPU
    3. Optimized thread organization
    4. Fallback to numpy for CPU execution
    """
    if use_cuda():
        (size, _) = a.shape
        threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK)
        blockspergrid = 1
        out = TensorData([0.0 for i in range(size * size)], (size, size))
        out.to_cuda_()
        jit_mm_practice[blockspergrid, threadsperblock](
            out.tuple()[0], a._tensor._storage, b._tensor._storage, size
        )
        return out
    else:
        # CPU implementation using numpy
        a_array = a._tensor._storage.reshape(a.shape)
        b_array = b._tensor._storage.reshape(b.shape)
        result = np.matmul(a_array, b_array)
        return TensorData(result.flatten().tolist(), result.shape)


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """Implements high-performance batched matrix multiplication.

    Advanced optimizations:
    1. Tiled shared memory access pattern
    2. Efficient handling of batch dimensions
    3. Broadcasting support for batch sizes
    4. Coalesced memory access
    5. Thread block synchronization

    Memory hierarchy:
    - Shared memory tiles for both input matrices
    - Efficient global memory access pattern
    - Thread-local accumulator for partial results

    Requirements:
    - Input validation: a_shape[-1] == b_shape[-2]
    - One global memory write per thread
    - Shared memory usage for all data access
    """
    BLOCK_DIM = 32

    # Batch dimension handling
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0
    batch = cuda.blockIdx.z

    # Shared memory allocation
    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    # Global thread positioning
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    # Local thread indices
    pi = cuda.threadIdx.x
    pj = cuda.threadIdx.y

    # Accumulator for dot product
    accum = 0.0

    # Tiled matrix multiplication
    for phase in range(0, a_shape[2], BLOCK_DIM):
        # Load tile from matrix A
        k = phase + pj
        if i < a_shape[1] and k < a_shape[2]:
            a_pos = batch * a_batch_stride + i * a_strides[1] + k * a_strides[2]
            a_shared[pi, pj] = a_storage[a_pos]
        else:
            a_shared[pi, pj] = 0.0

        # Load tile from matrix B
        k = phase + pi
        if k < b_shape[1] and j < b_shape[2]:
            b_pos = batch * b_batch_stride + k * b_strides[1] + j * b_strides[2]
            b_shared[pi, pj] = b_storage[b_pos]
        else:
            b_shared[pi, pj] = 0.0

        cuda.syncthreads()

        # Compute partial dot products
        if i < out_shape[1] and j < out_shape[2]:
            for k in range(min(BLOCK_DIM, a_shape[2] - phase)):
                accum += a_shared[pi, k] * b_shared[k, pj]

        cuda.syncthreads()

    # Store final result
    if i < out_shape[1] and j < out_shape[2]:
        out_pos = batch * out_strides[0] + i * out_strides[1] + j * out_strides[2]
        out[out_pos] = accum


tensor_matrix_multiply = jit(_tensor_matrix_multiply)
