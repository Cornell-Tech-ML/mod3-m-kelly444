import time
import numpy as np
from minitorch import tensor_ops
from minitorch import fast_ops

def benchmark_matmul(sizes=[2, 4, 8, 16, 32, 64]):
    naive_times = []
    fast_times = []
    
    for s in sizes:
        # Create random matrices
        a = np.random.randn(s, s)
        b = np.random.randn(s, s)
        
        # Time naive implementation
        start = time.time()
        tensor_ops.matrix_multiply(a, b)
        naive_times.append(time.time() - start)
        
        # Time fast implementation
        start = time.time()
        fast_ops._tensor_matrix_multiply(a, b)
        fast_times.append(time.time() - start)
    
    # Plot results
    import matplotlib.pyplot as plt
    plt.plot(sizes, naive_times, label='Naive')
    plt.plot(sizes, fast_times, label='Optimized')
    plt.xlabel('Matrix Size')
    plt.ylabel('Time (s)')
    plt.legend()
    plt.savefig('matmul_benchmark.png')