from bitmask import BitmaskArray
import numpy as np
import pytest

def test_constructor():
    arr = np.array([True, False, True, False, False])
    bma = BitmaskArray(arr)
    for index, x in enumerate(arr):
        assert bma[index] == x

@pytest.mark.parametrize("first_index,second_index", ([1, 2], [-3, -2]))
def test_settiem(first_index, second_index):
    arr = np.array([True, False, True, True])
    bma = BitmaskArray(arr)
    assert bma[first_index] == False
    assert bma[second_index] == True
    bma[first_index] = True
    bma[second_index] = False
    assert bma[first_index] == True
    assert bma[second_index] == False

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

"""
def test_buffer_protocol():
    arr = np.array([True, False, True, False, False])
    bma = BitmaskArray(arr)
    mv = memoryview(bma)
    for index, x in enumerate(arr):
        assert mv[index] == x
"""
