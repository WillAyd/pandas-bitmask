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

def test_and():
    arr = np.array([True, False, True, False, False])
    other = np.array([True, True, False, True, True])
    bma = BitmaskArray(arr)
    bma_other = BitmaskArray(other)
    result = bma & bma_other

    assert result[0]
    assert not result[1]
    assert not result[2]
    assert not result[3]
    assert not result[4]

def test_or():
    arr = np.array([True, False, True, False, False])
    other = np.array([True, True, False, True, True])
    bma = BitmaskArray(arr)
    bma_other = BitmaskArray(other)
    result = bma | bma_other

    assert result[0]
    assert result[1]
    assert result[2]
    assert result[3]
    assert result[4]

def test_xor():
    arr = np.array([True, False, True, False, False])
    other = np.array([True, True, False, True, True])
    bma = BitmaskArray(arr)
    bma_other = BitmaskArray(other)
    result = bma ^ bma_other

    assert not result[0]
    assert result[1]
    assert result[2]
    assert result[3]
    assert result[4]

def test_size():
    arr = np.array([True, False, True, False, False])
    bma = BitmaskArray(arr)

    assert bma.size == len(arr)

def test_nbytes():
    arr = np.array([True, False, True, False, False, True, True, True, True])
    bma = BitmaskArray(arr)

    assert bma.nbytes == 2

def test_any():
    arr = np.array([True, False, True, False, False, True, True, True, True])
    bma = BitmaskArray(arr)

    assert arr.any()

def test_all():
    arr = np.array([True, False, True, False, False, True, True, True, True])
    bma = BitmaskArray(arr)

    assert not arr.all()

"""
def test_buffer_protocol():
    arr = np.array([True, False, True, False, False])
    bma = BitmaskArray(arr)
    mv = memoryview(bma)
    for index, x in enumerate(arr):
        assert mv[index] == x
"""
