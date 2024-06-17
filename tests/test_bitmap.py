from bitmap import BitmaskArray
import numpy as np

def test_constructor():
    arr = np.array([True, False, True, False, False])
    bma = BitmaskArray(arr)
    for index, x in enumerate(arr):
        assert bma[index] == x

def test_length():
    arr = np.array([True, False, True, False, False])
    bma = BitmaskArray(arr)
    assert len(bma) == len(arr)

def test_invert():
    arr = np.array([True, False, True, False, False])
    bma = BitmaskArray(arr)
    inverted = ~bma
    for index, x in enumerate(arr):
        assert inverted[index] != x    

def test_buffer_protocol():
    arr = np.array([True, False, True, False, False])
    bma = BitmaskArray(arr)
    mv = memoryview(bma)
    for index, x in enumerate(arr):
        assert mv[index] == x
