import io
import pickle

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

@pytest.mark.parametrize(
    "arr,expected",
    [
        pytest.param(np.array([False, False]), bytes([0x0]), id="all_false"),
        pytest.param(np.array([True, False]), bytes([0x1]), id="first_true"),
        pytest.param(np.array([False, True]), bytes([0x2]), id="second_true"),
        pytest.param(np.array([True, True]), bytes([0x3]), id="all_true"),
        pytest.param(np.array([True, False] * 8), bytes([0x55, 0x55]), id="multibyte"),
    ],
)
def test_bytes(arr, expected):
    bma = BitmaskArray(arr)
    assert bma.bytes == expected

def test_dtype():
    arr = np.array([True, False, True, False, False, True, True, True, True])
    bma = BitmaskArray(arr)

    assert bma.dtype == "bool"

def test_any():
    arr = np.array([True, False, True, False, False, True, True, True, True])
    bma = BitmaskArray(arr)

    assert arr.any()

def test_all():
    arr = np.array([True, False, True, False, False, True, True, True, True])
    bma = BitmaskArray(arr)

    assert not arr.all()

def test_sum():
    arr = np.array([True, False, True, False, False, True, True, True, True])
    bma = BitmaskArray(arr)

    assert bma.sum() == 6

def test_copy():
    arr = np.array([True, False, True, False, False, True, True, True, True])
    bma = BitmaskArray(arr)
    copied = bma.copy()

    assert copied is not bma
    for i in range(len(arr)):
        assert bma[i] == copied[i]

def test_numpy_implicit_conversion():
    arr = np.array([True, False, True, False, False])
    bma = BitmaskArray(arr)

    arr2 = np.array(bma)
    assert (arr == arr2).all()

def test_pickle_roundtrip():
    arr = np.array([True, False, True, False, False])
    bma = BitmaskArray(arr)

    buf = io.BytesIO()
    pickle.dump(bma, buf)

    buf.seek(0)
    bma2 = pickle.load(buf)

    for i in range(len(arr)):
        assert bma[i] == bma2[i]
