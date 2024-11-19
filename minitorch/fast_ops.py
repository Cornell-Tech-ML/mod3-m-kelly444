from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, Any

import numpy as np
from numba import prange
from numba import njit as _njit

from .tensor_data import (
    MAX_DIMS,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

if TYPE_CHECKING:
    from typing import Callable, Optional

    from .tensor import Tensor
    from .tensor_data import Shape, Storage, Strides

# To disable JIT compilation during testing, use: NUMBA_DISABLE_JIT=1 pytest tests/ -m task3_1

# JIT compilation is applied to optimize tensor operations
# Any modifications to these functions must comply with NUMBA's restrictions
Fn = TypeVar("Fn")


def njit(fn: Fn, **kwargs: Any) -> Fn:
    """Wraps functions with NUMBA's JIT compiler for performance optimization."""
    return _njit(inline="always", **kwargs)(fn)  # type: ignore


to_index = njit(to_index)
index_to_position = njit(index_to_position)
broadcast_index = njit(broadcast_index)


class FastOps(TensorOps):
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """Applies an element-wise operation to a tensor.

        Creates a JIT-compiled version of the mapping function for faster execution.
        Returns a closure that handles the actual tensor operation.
        """
        f = tensor_map(njit(fn))

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)
            f(*out.tuple(), *a.tuple())
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        """Combines two tensors element-wise using the provided function.

        Handles broadcasting and creates an optimized version of the operation.
        """
        f = tensor_zip(njit(fn))

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            f(*out.tuple(), *a.tuple(), *b.tuple())
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """Reduces a tensor along a specified dimension using the given function.

        The operation maintains the dimension with size 1 for broadcasting compatibility.
        """
        f = tensor_reduce(njit(fn))

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = 1
            out = a.zeros(tuple(out_shape))
            out._tensor._storage[:] = start
            f(*out.tuple(), *a.tuple(), dim)
            return out

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """Performs batched matrix multiplication between two tensors.

        Supports broadcasting across batch dimensions and handles both 2D and 3D inputs.
        The last two dimensions are treated as matrix dimensions, where:
        - out[n, i, j] = sum_k(a[n, i, k] * b[n, k, j])

        Requirements:
        - Inner dimensions must match (a.shape[-1] == b.shape[-2])
        - Batch dimensions must be broadcastable
        """
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        tensor_matrix_multiply(*out.tuple(), *a.tuple(), *b.tuple())

        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """Creates a parallelized element-wise mapping operation.

    Performance optimizations:
    1. Parallel execution of the main loop
    2. Thread-local index buffers
    3. Efficient stride handling for aligned tensors

    The function processes each element independently, making it ideal for parallelization.
    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        for i in prange(len(out)):
            # Thread-safe index computation
            out_index = np.empty(MAX_DIMS, np.int32)
            in_index = np.empty(MAX_DIMS, np.int32)

            to_index(i, out_shape, out_index)
            broadcast_index(out_index, out_shape, in_shape, in_index)
            o = index_to_position(out_index, out_strides)
            j = index_to_position(in_index, in_strides)
            out[o] = fn(in_storage[j])

    return njit(_map, parallel=True)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """Creates a parallelized element-wise operation between two tensors.

    Performance optimizations:
    1. Parallel processing of output elements
    2. Thread-local index management
    3. Efficient stride handling for aligned inputs

    Handles broadcasting automatically for mismatched shapes.
    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        for i in prange(len(out)):
            # Per-thread index buffers
            out_index = np.empty(MAX_DIMS, np.int32)
            a_index = np.empty(MAX_DIMS, np.int32)
            b_index = np.empty(MAX_DIMS, np.int32)

            to_index(i, out_shape, out_index)
            o = index_to_position(out_index, out_strides)
            broadcast_index(out_index, out_shape, a_shape, a_index)
            j = index_to_position(a_index, a_strides)
            broadcast_index(out_index, out_shape, b_shape, b_index)
            k = index_to_position(b_index, b_strides)
            out[o] = fn(a_storage[j], b_storage[k])

    return njit(_zip, parallel=True)  # type: ignore


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """Creates a parallelized reduction operation along a specified dimension.

    Performance optimizations:
    1. Parallel processing of output elements
    2. Thread-local index management
    3. Minimized function calls in the inner loop

    The reduction maintains dimension for later broadcasting compatibility.
    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
    ) -> None:
        out_index = np.zeros(MAX_DIMS, np.int32)
        reduce_size = a_shape[reduce_dim]

        for i in prange(len(out)):
            # Per-thread index management
            out_index = np.empty(MAX_DIMS, np.int32)
            local_index = np.empty(MAX_DIMS, np.int32)

            to_index(i, out_shape, out_index)
            o = index_to_position(out_index, out_strides)

            for j in range(len(out_shape)):
                local_index[j] = out_index[j]

            # Reduction loop
            for s in range(reduce_size):
                local_index[reduce_dim] = s
                j = index_to_position(local_index, a_strides)
                out[o] = fn(out[o], a_storage[j])

    return njit(_reduce, parallel=True)  # type: ignore


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """Implements efficient batched matrix multiplication for tensors.

    Performance optimizations:
    1. Parallel processing of output rows
    2. Minimized memory access patterns
    3. Cache-friendly inner loop
    4. Direct stride manipulation

    The implementation handles:
    - Batched operations with broadcasting
    - Efficient memory access patterns
    - Parallel computation across output elements
    """
    # Handle batch dimension strides
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0
    blocks = a_shape[-1]

    # Parallel processing of output matrix
    for row_i in prange(0, out_shape[0]):
        for col_j in range(0, out_shape[1]):
            for block_k in range(0, out_shape[2]):
                # Compute base positions
                row_s = row_i * a_batch_stride + col_j * a_strides[1]
                col_s = row_i * b_batch_stride + block_k * b_strides[2]

                temp = 0.0

                # Matrix multiplication inner loop
                for _ in range(0, blocks):
                    temp += a_storage[row_s] * b_storage[col_s]
                    row_s += a_strides[-1]
                    col_s += b_strides[-2]

                # Store result
                out[
                    row_i * out_strides[0]
                    + col_j * out_strides[1]
                    + block_k * out_strides[2]
                ] = temp


tensor_matrix_multiply = njit(_tensor_matrix_multiply, parallel=True)
assert tensor_matrix_multiply is not None
