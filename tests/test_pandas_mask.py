import io
import operator
import pickle

from pandas_mask import PandasMaskArray
import numpy as np
import pytest

def test_constructor():
    arr = np.array([True, False, True, False, False])
    bma = PandasMaskArray(arr)
    for index, x in enumerate(arr):
        assert bma[index] == x

def test_constructor_from_another_mask():
    arr = np.array([True, False, True, False, False])
    bma = PandasMaskArray(arr)

    bma2 = PandasMaskArray(bma)
    for index, x in enumerate(arr):
        assert bma2[index] == x

    # should copy - updates to original bma should not propogate
    bma[1] = True
    assert bma[1]
    assert not bma2[1]

def test_getitem_bounds_raise():
    arr = np.array([True, False, True, True])
    bma = PandasMaskArray(arr)

    bma[3]
    with pytest.raises(IndexError):
        bma[4]

    bma[-4]
    with pytest.raises(IndexError):
        bma[-5]

def test_getitem_list():
    arr = np.array([True, False, True, True])
    bma = PandasMaskArray(arr)

    idxer = [3, 1, 2, 1, 0]
    result = bma[idxer]

    assert len(result) == 5
    assert result[0]
    assert not result[1]
    assert result[2]
    assert not result[3]
    assert result[4]

def test_getitem_ndarray_bools():
    arr = np.array([True, False, True, True])
    bma = PandasMaskArray(arr)

    idxer = np.array([True, True, False, False])
    result = bma[idxer]

    assert len(result) == 2
    assert result[0]
    assert not result[1]

def test_getitem_ndarray_ints():
    arr = np.array([True, False, True, True])
    bma = PandasMaskArray(arr)

    idxer = np.array([3, 1, 2, 1, 0])
    result = bma[idxer]

    idxer = [3, 1, 2, 1, 0]

    result = bma[idxer]
    assert len(result) == 5
    assert result[0]
    assert not result[1]
    assert result[2]
    assert not result[3]
    assert result[4]

def test_getitem_ndarray_raises():
    arr = np.array([True, False, True, True])
    bma = PandasMaskArray(arr)

    idxer = np.array([True, True, True])
    with pytest.raises(ValueError, match="must be same size"):
        bma[idxer]

def test_getitem_slice():
    arr = np.array([True, False, True, True])
    bma = PandasMaskArray(arr)

    result = bma[1:]
    assert len(result) == 3
    assert not result[0]
    assert result[1]
    assert result[2]

@pytest.mark.parametrize("first_index,second_index", ([1, 2], [-3, -2]))
def test_settiem(first_index, second_index):
    arr = np.array([True, False, True, True])
    bma = PandasMaskArray(arr)
    assert bma[first_index] == False
    assert bma[second_index] == True
    bma[first_index] = True
    bma[second_index] = False
    assert bma[first_index] == True
    assert bma[second_index] == False

def test_length():
    arr = np.array([True, False, True, False, False])
    bma = PandasMaskArray(arr)
    assert len(bma) == len(arr)

def test_invert():
    arr = np.array([True, False, True, False, False])
    bma = PandasMaskArray(arr)
    inverted = ~bma
    for index, x in enumerate(arr):
        assert inverted[index] != x

def test_and():
    arr = np.array([True, False, True, False, False])
    other = np.array([True, True, False, True, True])
    bma = PandasMaskArray(arr)
    bma_other = PandasMaskArray(other)
    result = bma & bma_other

    assert result[0]
    assert not result[1]
    assert not result[2]
    assert not result[3]
    assert not result[4]

def test_and_numpy():
    arr = np.array([True, False, True, False, False])
    other = np.array([True, True, False, True, True])
    bma = PandasMaskArray(arr)
    result = bma & other

    assert result[0]
    assert not result[1]
    assert not result[2]
    assert not result[3]
    assert not result[4]

def test_or():
    arr = np.array([True, False, True, False, False])
    other = np.array([True, True, False, True, True])
    bma = PandasMaskArray(arr)
    bma_other = PandasMaskArray(other)
    result = bma | bma_other

    assert result[0]
    assert result[1]
    assert result[2]
    assert result[3]
    assert result[4]

def test_or_numpy():
    arr = np.array([True, False, True, False, False])
    other = np.array([True, True, False, True, True])
    bma = PandasMaskArray(arr)
    result = bma | other

    assert result[0]
    assert result[1]
    assert result[2]
    assert result[3]
    assert result[4]

def test_xor():
    arr = np.array([True, False, True, False, False])
    other = np.array([True, True, False, True, True])
    bma = PandasMaskArray(arr)
    bma_other = PandasMaskArray(other)
    result = bma ^ bma_other

    assert not result[0]
    assert result[1]
    assert result[2]
    assert result[3]
    assert result[4]

def test_xor_numpy():
    arr = np.array([True, False, True, False, False])
    other = np.array([True, True, False, True, True])
    bma = PandasMaskArray(arr)
    result = bma ^ other

    assert not result[0]
    assert result[1]
    assert result[2]
    assert result[3]
    assert result[4]

@pytest.mark.parametrize("op", [operator.and_, operator.or_, operator.xor])
def test_binop_raises(op):
    arr = np.array([True, False, True, False, False])
    bma = PandasMaskArray(arr)

    with pytest.raises(TypeError):
        result = op(bma, "foo")

def test_size():
    arr = np.array([True, False, True, False, False])
    bma = PandasMaskArray(arr)

    assert bma.size == len(arr)

def test_nbytes():
    arr = np.array([True, False, True, False, False, True, True, True, True])
    bma = PandasMaskArray(arr)

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
    bma = PandasMaskArray(arr)
    assert bma.bytes == expected

def test_dtype():
    arr = np.array([True, False, True, False, False, True, True, True, True])
    bma = PandasMaskArray(arr)

    assert bma.dtype == "bool"

def test_any():
    arr = np.array([True, False, True, False, False, True, True, True, True])
    bma = PandasMaskArray(arr)

    assert arr.any()

def test_all():
    arr = np.array([True, False, True, False, False, True, True, True, True])
    bma = PandasMaskArray(arr)

    assert not arr.all()

def test_sum():
    arr = np.array([True, False, True, False, False, True, True, True, True])
    bma = PandasMaskArray(arr)

    assert bma.sum() == 6

def test_copy():
    arr = np.array([True, False, True, False, False, True, True, True, True])
    bma = PandasMaskArray(arr)
    copied = bma.copy()

    assert copied is not bma
    for i in range(len(arr)):
        assert bma[i] == copied[i]

def test_numpy_implicit_conversion():
    arr = np.array([True, False, True, False, False])
    bma = PandasMaskArray(arr)

    arr2 = np.array(bma)
    assert (arr == arr2).all()

def test_pickle_roundtrip():
    arr = np.array([True, False, True, False, False])
    bma = PandasMaskArray(arr)

    buf = io.BytesIO()
    pickle.dump(bma, buf)

    buf.seek(0)
    bma2 = pickle.load(buf)

    for i in range(len(arr)):
        assert bma[i] == bma2[i]

def test_iter():
    arr = np.array([True, False, True, False, False])
    bma = PandasMaskArray(arr)

    itr = iter(bma)
    assert next(itr)
    assert not next(itr)
    assert next(itr)
    assert not next(itr)
    assert not next(itr)

    with pytest.raises(StopIteration):
        next(itr)

def test_shape():
    arr = np.array([True, False, True, False, False])
    bma = PandasMaskArray(arr)

    assert bma.shape == tuple((5,))
