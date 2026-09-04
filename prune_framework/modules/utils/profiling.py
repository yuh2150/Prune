import time
import torch
from contextlib import contextmanager


@contextmanager
def timer(name: str = "Operation"):
    t0 = time.perf_counter()
    yield
    t1 = time.perf_counter()
    print(f"[{name}] Completed in {(t1 - t0) * 1000.0:.2f} ms")
