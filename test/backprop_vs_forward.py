import time
import random
import sys
import os
# ensure project root on path so we can import micrograd files
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from micrograd.engine import Value
from micrograd.forward_engine import F_Value

def build_expr_values(xs):
    # y = sum( (x_i * x_{i+1}) ** 2 + x_i ) with a ReLU-like nonlinearity
    n = len(xs)
    y = Value(0.0)
    for i in range(n):
        a = xs[i]
        b = xs[i] + 3
        tmp = (a * b) ** 2
        y = y + tmp + a.relu()
    return y

def build_expr_duals(xs):
    n = len(xs)
    y = F_Value(0.0)
    for i in range(n):
        a = xs[i]
        b = xs[i] + 3
        tmp = (a * b) ** 2
        y = y + tmp + a.relu()
    return y

def benchmark(n_inputs, n_repeats=1):
    # create random inputs
    vals = [random.uniform(-1, 1) for _ in range(n_inputs)]

    # Reverse-mode: single backward computes grads for all inputs
    t_backward = 0.0
    for _ in range(n_repeats):
        xs = [Value(v) for v in vals]
        y = build_expr_values(xs)
        t0 = time.perf_counter()
        y.backward()
        t1 = time.perf_counter()
        t_backward += (t1 - t0)
    t_backward /= n_repeats
    
    # Forward-mode: need one forward pass per input to get full gradient
    t_forward = 0.0
    for _ in range(n_repeats):
        # start timer
        t0 = time.perf_counter()
        grads = []
        for i in range(n_inputs):
            # build f_values with dot=1 for i-th input (seed)
            xs = [F_Value(vals[j], 1.0 if j == i else 0.0) for j in range(n_inputs)]
            y = build_expr_duals(xs)
            grads.append(y.dot)
        # stop timer
        t1 = time.perf_counter()
        t_forward += (t1 - t0)
    t_forward /= n_repeats

    return {
        "n_inputs": n_inputs,
        "backward_s": t_backward,
        "forward_s": t_forward
    }

if __name__ == "__main__":
    sizes = [10, 50, 100, 200, 400]
    print("n\treverse_backward(s)\tforward(s)\tSpeedup")
    for n in sizes:
        r = benchmark(n, n_repeats=3)
        speedup = r['forward_s'] / r['backward_s'] if r['forward_s'] > 0 else 0
        print(f"{r['n_inputs']}\t\t{r['backward_s']:.6f}\t{r['forward_s']:.6f}\t{speedup:.2f}x")