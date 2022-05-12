import functools
from time import perf_counter
from typing import Callable


def timeit(func: Callable) -> Callable:

    @functools.wraps(func)
    def wrapper(*args, **kwargs):

        start = perf_counter()
        result = func(*args, **kwargs)
        wrapper.time = perf_counter() - start

        return result

    wrapper.time = 0
    
    return wrapper