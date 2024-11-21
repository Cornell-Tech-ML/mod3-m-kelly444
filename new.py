import time
from minitorch import fast_ops
from minitorch import tensor_ops
import numpy as np

def time_matmul(size):
    # Create test matrices as 3D tensors to match implementation
    # Adding batch dimension of 1 at the front
    a = np.random.randn(1, size, size).astype(np.float32)
    b = np.random.randn(1, size, size).astype(np.float32)
    out = np.zeros((1, size, size)).astype(np.float32)
    
    # Setup shapes and strides
    out_shape = (1, size, size)
    a_shape = (1, size, size)
    b_shape = (1, size, size)
    
    # Calculate strides for 3D arrays
    out_strides = (size * size, size, 1)
    a_strides = (size * size, size, 1)
    b_strides = (size * size, size, 1)
    
    # Time parallel CPU implementation
    start = time.time()
    try:
        fast_ops._tensor_matrix_multiply(
            out.reshape(-1),  # out storage
            out_shape,
            out_strides,
            a.reshape(-1),   # a storage
            a_shape,
            a_strides,
            b.reshape(-1),   # b storage
            b_shape,
            b_strides
        )
        parallel_time = (time.time() - start) * 1000
    except Exception as e:
        print(f"Error in parallel implementation: {e}")
        parallel_time = float('inf')
    
    # Time baseline implementation
    start = time.time()
    try:
        tensor_ops.matrix_multiply(
            out.reshape(-1),
            out_shape,
            out_strides,
            a.reshape(-1),
            a_shape,
            a_strides,
            b.reshape(-1),
            b_shape,
            b_strides
        )
        baseline_time = (time.time() - start) * 1000
    except Exception as e:
        print(f"Error in baseline implementation: {e}")
        baseline_time = float('inf')
    
    return parallel_time, baseline_time

# Test different sizes
sizes = [16, 32, 64, 128]  # Starting with smaller sizes for testing
trials = 3
results = []

for size in sizes:
    parallel_times = []
    baseline_times = []
    
    print(f"\nTesting size {size}x{size}")
    for t in range(trials):
        try:
            parallel_t, baseline_t = time_matmul(size)
            if parallel_t != float('inf') and baseline_t != float('inf'):
                parallel_times.append(parallel_t)
                baseline_times.append(baseline_t)
                print(f"Trial {t+1}: Parallel: {parallel_t:.2f}ms, Baseline: {baseline_t:.2f}ms")
        except Exception as e:
            print(f"Error in trial {t+1}: {e}")
            continue
    
    if parallel_times and baseline_times:
        results.append({
            'size': size,
            'parallel': np.mean(parallel_times),
            'baseline': np.mean(baseline_times)
        })
        print(f"Average for size {size}: Parallel: {np.mean(parallel_times):.2f}ms, Baseline: {np.mean(baseline_times):.2f}ms")
    else:
        print(f"No valid results for size {size}")

print("\nFinal Results:")
for r in results:
    print(f"Size {r['size']}x{r['size']}: Parallel = {r['parallel']:.2f}ms, Baseline = {r['baseline']:.2f}ms")