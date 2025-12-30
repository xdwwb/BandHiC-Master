import operator
import sys
import os
import numbers

# Pytest-based test suite for band_hic_matrix
import pytest
import numpy as np
from scipy.sparse import coo_matrix
from bandhic import band_hic_matrix, straw_chr
import types
import copy
import bandhic as bh


@pytest.fixture(
    params=[
        # Original example without mask
        {
            "row": np.array([0, 1, 2, 3]),
            "col": np.array([0, 1, 2, 3]),
            "data": np.array([1, 2, 3, 4]),
            "diag_num": 2,
            "mask": None,
            "expected_array": np.diag([1, 2, 3, 4]),
        },
        # New example with mask
        {
            "row": np.array([0, 1, 2, 3]),
            "col": np.array([0, 1, 2, 3]),
            "data": np.array([1, 2, 3, 4]),
            "diag_num": 2,
            "mask": np.array(
                [[False, True], [False, False], [True, False], [False, True]]
            ),
            "expected_array": np.ma.array(
                np.diag([1, 2, 3, 4]),
                mask=[
                    [False, True, False, False],
                    [True, False, False, False],
                    [False, False, True, False],
                    [False, False, False, False],
                ],
            ),
        },
        # Original example without mask
        {
            "row": np.array([0, 1, 2, 3, 0, 1, 2]),
            "col": np.array([0, 1, 2, 3, 1, 2, 3]),
            "data": np.array([1, 2, 3, 4, 5, 6, 7]),
            "diag_num": 2,
            "mask": None,
            "expected_array": np.array(
                [[1, 5, 0, 0], [5, 2, 6, 0], [0, 6, 3, 7], [0, 0, 7, 4]]
            ),
        },
        # New example with mask
        {
            "row": np.array([0, 1, 2, 3, 0, 1, 2]),
            "col": np.array([0, 1, 2, 3, 1, 2, 3]),
            "data": np.array([1, 2, 3, 4, 5, 6, 7]),
            "diag_num": 2,
            "mask": np.array(
                [[False, True], [False, False], [True, False], [False, True]]
            ),
            "expected_array": np.ma.array(
                [[1, 5, 0, 0], [5, 2, 6, 0], [0, 6, 3, 7], [0, 0, 7, 4]],
                mask=[
                    [False, True, False, False],
                    [True, False, False, False],
                    [False, False, True, False],
                    [False, False, False, False],
                ],
            ),
        },
    ]
)
def example_band_matrix(request):
    params = request.param
    if "row" in params:
        hic_coo = coo_matrix(
            (params["data"], (params["row"], params["col"])), shape=(4, 4)
        )
        mat = band_hic_matrix(hic_coo, diag_num=params["diag_num"])
        if params["mask"] is not None:
            mat.mask = params["mask"].copy()
        return mat.copy(), copy.deepcopy(params["expected_array"])
    elif "path" in params:
        mat = bh.straw_chr(
            params["path"],
            params["chrom"],
            params["resolution"],
            params["diag_num"],
        )
        mat_dense = mat.todense()
        row_sum = mat_dense.sum(axis=0)
        mat.add_mask_row_col(row_sum == 0)
        mat_dense = np.ma.masked_array(mat_dense, mask=False, fill_value=0)
        mat_dense[row_sum == 0, :] = np.ma.masked
        mat_dense[:, row_sum == 0] = np.ma.masked
        return mat.copy(), copy.deepcopy(mat_dense)


def assert_result_equal(a, b):
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        if np.array_equal(a, b):
            return True
        else:
            raise AssertionError(
                "Arrays do not match, arrays: {} vs {}".format(a, b)
            )
    elif isinstance(a, numbers.Number) and isinstance(b, numbers.Number):
        if a == b:
            return True
        else:
            raise AssertionError(
                "Numbers do not match, numbers: {} vs {}".format(a, b)
            )
    elif isinstance(a, np.ma.MaskedArray) and isinstance(b, np.ma.MaskedArray):
        if np.array_equal(a.mask, b.mask):
            if np.array_equal(a.data[~a.mask], b.data[~b.mask]):
                return True
            else:
                raise AssertionError(
                    "Data in masked arrays do not match, masked arrays: {} vs {}".format(
                        a, b
                    )
                )
        else:
            raise AssertionError(
                "Masks in masked arrays do not match, masks: {} vs {}".format(
                    a.mask, b.mask
                )
            )
    elif a == b == np.ma.masked:
        return True
    else:
        raise AssertionError(
            "The types of the two objects are not comparable, types: {} vs {}, a: {} vs b: {}".format(
                type(a), type(b), a, b
            )
        )


def test_initialization_and_shape(example_band_matrix):
    mat, expected_array = example_band_matrix
    assert mat.shape == expected_array.shape
    # Only main diagonal filled
    # np.testing.assert_array_equal(
    #     mat.data, np.array([[1, 0], [2, 0], [3, 0], [4, 0]])
    # )


def test_set_and_get_values(example_band_matrix):
    mat, expected_array = example_band_matrix
    mat.set_values([0, 1], [1, 2], [5, 6])
    assert mat.data[0, 1] == 5
    assert mat.data[1, 1] == 6
    # get_values returns values for both filled and zero
    vals = mat.get_values([0, 1], [0, 1])
    np.testing.assert_array_equal(vals, np.array([1, 2]))


def test_set_mask_and_masking(example_band_matrix):
    mat, expected_array = example_band_matrix
    mat.add_mask([0, 1], [1, 2])
    assert mat.mask[0, 1]
    assert mat.mask[1, 1]
    # After masking, get_values returns masked array
    arr = mat.get_values([0, 1], [1, 2])
    assert np.all(arr.mask)


def test_get_and_setitem(example_band_matrix):
    mat, expected_array = example_band_matrix
    mat.set_values([0, 1], [1, 2], [5, 6])
    expected_array[[0, 1], [1, 2]] = [5, 6]
    expected_array[[1, 2], [0, 1]] = [5, 6]
    # __getitem__ for single index
    # assert_result_equal(mat[0, 1], expected_array[0, 1])
    # __getitem__ for slicing (returns ndarray)
    arr = mat[0:2]
    assert isinstance(arr, band_hic_matrix)
    # np.testing.assert_array_equal(arr.todense(), expected_array[0:2, 0:2])
    # __setitem__
    mat[0, 1] = 7
    expected_array[0, 1] = 7
    expected_array[1, 0] = 7
    # assert mat[0, 1] == expected_array[0, 1]
    # assert_result_equal(mat.todense(), expected_array)


def test_diag_and_set_diag(example_band_matrix):
    mat, expected_array = example_band_matrix
    np.testing.assert_array_equal(mat.diag(0), np.array([1, 2, 3, 4]))
    mat.set_diag(1, [10, 11, 12])
    np.testing.assert_array_equal(mat.data[:3, 1], [10, 11, 12])
    np.testing.assert_array_equal(mat.diag(1), np.array([10, 11, 12]))


def test_todense_and_tocoo(example_band_matrix):
    mat, expected_array = example_band_matrix
    dense = mat.todense()
    expected = expected_array
    np.testing.assert_array_equal(dense, expected)
    coo = mat.tocoo()
    # Only main diagonal has data
    # if mat.mask is None:
    #     np.testing.assert_array_equal(coo.row, np.array([0, 1, 2, 3]))
    #     np.testing.assert_array_equal(coo.col, np.array([0, 1, 2, 3]))
    #     np.testing.assert_array_equal(coo.data, np.array([1, 2, 3, 4]))
    # else:
    #     np.testing.assert_array_equal(coo.row, np.array([0, 1, 3]))
    #     np.testing.assert_array_equal(coo.col, np.array([0, 1, 3]))
    #     np.testing.assert_array_equal(coo.data, np.array([1, 2, 4]))


def test_slicing_returns_band_matrix(example_band_matrix):
    mat, expected_array = example_band_matrix
    sub = mat[1:3]
    assert isinstance(sub, band_hic_matrix)
    assert sub.shape == (2, 2)
    np.testing.assert_array_equal(sub.todense(), expected_array[1:3, 1:3])


def test_extract_row(example_band_matrix):
    mat, expected_array = example_band_matrix
    row = mat.extract_row(0, extract_out_of_band=True)
    assert_result_equal(row, expected_array[0])

    row = mat.extract_row(1, extract_out_of_band=True)
    assert_result_equal(row, expected_array[1])

    row = mat.extract_row(3, extract_out_of_band=True)
    assert_result_equal(row, expected_array[3])

    row = mat.extract_row(0, extract_out_of_band=False)
    assert_result_equal(row, expected_array[0, :2])

    row = mat.extract_row(1, extract_out_of_band=False)
    assert_result_equal(row, expected_array[1, :3])

    row = mat.extract_row(3, extract_out_of_band=False)
    assert_result_equal(row, expected_array[3, 2:])


@pytest.mark.parametrize(
    "axis,expected",
    [
        (None, 0),  # min of all
        ("row", np.array([0, 0, 0, 0])),
        ("col", np.array([0, 0, 0, 0])),
        ("diag", np.array([1, 0])),
    ],
)
def test_min_reduction(example_band_matrix, axis, expected):
    mat, expected_array = example_band_matrix
    result = mat.min(axis=axis)
    if axis == "col":
        axis = 1
    elif axis == "row":
        axis = 0
    if axis != "diag":
        assert_result_equal(result, expected_array.min(axis=axis))
    else:
        if mat.mask is None:
            assert_result_equal(
                result,
                np.array(
                    [
                        np.diag(expected_array, k=i).min()
                        for i in range(mat.diag_num)
                    ]
                ),
            )
        else:
            assert_result_equal(
                result,
                np.array(
                    [
                        np.ma.diag(expected_array, k=i).min()
                        for i in range(mat.diag_num)
                    ]
                ),
            )


@pytest.mark.parametrize("axis", [(None), (0), (1), ("diag")])  # max of all
def test_max_reduction(example_band_matrix, axis):
    mat, expected_array = example_band_matrix
    if axis != "diag":
        result = mat.max(axis=axis)
        assert_result_equal(result, expected_array.max(axis=axis))
    else:
        if mat.mask is None:
            assert_result_equal(
                mat.max(axis=axis),
                np.array(
                    [
                        np.diag(expected_array, k=i).max()
                        for i in range(mat.diag_num)
                    ]
                ),
            )
        else:
            assert_result_equal(
                mat.max(axis=axis),
                np.array(
                    [
                        np.ma.diag(expected_array, k=i).max()
                        for i in range(mat.diag_num)
                    ]
                ),
            )


def test_mean_sum_prod_var_std_ptp(example_band_matrix):
    mat, mat_expect = example_band_matrix
    # Only main diagonal is nonzero
    assert mat.mean() == mat_expect.mean()
    assert mat.sum() == mat_expect.sum()
    assert mat.prod() == np.prod(mat_expect)  # because zeros in off-diagonal
    assert mat.var() == mat_expect.var(ddof=0)
    assert mat.std() == mat_expect.std(ddof=0)
    assert mat.ptp() == mat_expect.ptp()
    # Axis reductions
    rows = mat.mean(axis="row")
    np.testing.assert_array_equal(rows, mat_expect.mean(axis=0))
    diags = mat.mean(axis="diag")
    if mat.mask is None:
        np.testing.assert_array_equal(
            diags,
            np.array(
                [np.diag(mat_expect, k=i).mean() for i in range(mat.diag_num)]
            ),
        )
    else:
        # Masked array mean
        np.testing.assert_array_equal(
            diags,
            np.array(
                [
                    np.ma.diag(mat_expect, k=i).mean()
                    for i in range(mat.diag_num)
                ]
            ),
        )


def test_all_any(example_band_matrix):
    mat, expected_array = example_band_matrix
    # Only main diagonal is nonzero, so not all positive
    assert not mat.all()
    assert mat.any()
    # Make all entries > 0
    mat.data[:] = 1
    mat.mask = None
    assert mat.all(banded_only=True)
    assert mat.any(banded_only=True)


def test_banded_count(example_band_matrix):
    mat, expected_array = example_band_matrix
    # Only main diagonal is nonzero
    if mat.mask is None:
        assert mat.count_masked() == 0
    else:
        assert mat.count_masked() == 3


def test_contains(example_band_matrix):
    mat, expected_array = example_band_matrix
    assert 2 in mat
    assert 9999999999 not in mat


def test_copy_and_astype(example_band_matrix):
    mat, expected_array = example_band_matrix
    m2 = mat.copy()
    assert m2.shape == mat.shape
    m2.astype(np.float32)
    assert m2.data.dtype == np.float32


def test_clip_and_fill(example_band_matrix):
    mat, expected_array = example_band_matrix
    mat.clip(2, 4)
    assert np.all(mat.data <= 4)
    mat2 = mat.filled(5)
    assert mat2.mask is None


def test_filled_and_reset_mask(example_band_matrix):
    mat, expected_array = example_band_matrix
    mat.add_mask([0, 1], [1, 1])
    mat2 = mat.filled(99)
    assert mat2.data[0, 1] == 99
    assert mat2.data[1, 0] == 99
    mat.clear_mask()
    assert mat.mask is None


def test_drop_mask_and_remove_mask(example_band_matrix):
    mat, expected_array = example_band_matrix
    mat.add_mask([0], [1])
    mat.clear_mask()
    assert mat.mask is None
    mat.add_mask([0], [1])
    mat.unmask(([0], [1]))
    assert not mat.mask[0, 1]


def test_iterrows_and_itercols(example_band_matrix):
    mat, expected_array = example_band_matrix
    rows = list(mat.iterrows())
    assert len(rows) == 4
    cols = list(mat.itercols())
    assert len(cols) == 4


def test_iterwindows(example_band_matrix):
    mat, expected_array = example_band_matrix
    wins = list(mat.iterwindows(2))
    assert all(isinstance(w, band_hic_matrix) for w in wins)
    assert wins[0].shape == (2, 2)


# def test_band_multiply(example_band_matrix):
#     mat, expected_array = example_band_matrix
#     # Multiply by itself
#     result = mat.band_multiply(mat)
#     assert isinstance(result, band_hic_matrix)
#     assert result.shape == mat.shape


def test_bool_and_array(example_band_matrix):
    mat = band_hic_matrix(np.array([[1]]), diag_num=1)
    assert bool(mat)
    mat[0, 0] = 0
    assert not bool(mat)
    arr = np.array(mat)
    np.testing.assert_array_equal(arr, mat.data)
    # Test ValueError for non-1x1
    mat2, expected_array = example_band_matrix
    with pytest.raises(ValueError):
        bool(mat2)


def test_repr_and_str(example_band_matrix):
    mat, expected_array = example_band_matrix
    r = repr(mat)
    s = str(mat)
    assert "band_hic_matrix" in r
    assert "band_hic_matrix" in s


def test_array_function_dispatch(example_band_matrix):
    mat, expected_array = example_band_matrix
    # np.sum should call band_hic_matrix.sum
    assert np.sum(mat) == mat.sum()
    assert np.min(mat) == mat.min()
    assert np.prod(mat) == mat.prod()
    # np.mean, np.var, np.std, np.ptp, np.all, np.any
    assert np.mean(mat) == mat.mean()
    assert np.var(mat) == mat.var()
    assert np.std(mat) == mat.std()
    assert np.ptp(mat) == mat.ptp()
    assert np.all(mat) == mat.all()
    assert np.any(mat) == mat.any()


def test_hash(example_band_matrix):
    mat, expected_array = example_band_matrix
    # Should not raise
    hash(mat)


def test_set_diag_errors(example_band_matrix):
    mat, expected_array = example_band_matrix
    with pytest.raises(ValueError):
        mat.set_diag(-1, [1])
    with pytest.raises(ValueError):
        mat.set_diag(2, [1, 2, 3, 4])  # diag_num=2, so k=2 is out of range
    with pytest.raises(ValueError):
        mat.set_diag(1, [1])  # length mismatch


def test_invalid_initialization():
    arr = np.eye(4)
    # diag_num > bin_num

    # non-square array
    with pytest.raises(ValueError):
        band_hic_matrix(np.zeros((3, 4)), diag_num=2)
    # 1D array
    with pytest.raises(ValueError):
        band_hic_matrix(np.ones(4), diag_num=1)


def test_memory_usage(example_band_matrix):
    mat, expected_array = example_band_matrix
    sz = mat.memory_usage()
    assert isinstance(sz, int)
    assert sz > 0


def test_iter(example_band_matrix):
    mat, expected_array = example_band_matrix
    # __iter__ yields diagonals
    diags = list(iter(mat))
    assert len(diags) == mat.diag_num
    for k, d in enumerate(diags):
        np.testing.assert_array_equal(d, mat.diag(k))


# def test_convolve2d(example_band_matrix):
#     mat, expected_array = example_band_matrix
#     kernel = np.ones((2, 2))
#     conv = mat.convolve2d(kernel)
#     assert isinstance(conv, np.ndarray) or (hasattr(conv, 'shape'))
#     assert conv.shape == mat.shape

import tempfile
import os


def test_dump_and_load(example_band_matrix):
    mat, expected_array = example_band_matrix
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "test_bandhic_dump.npz")
        mat.dump(fname)
        # Create a new instance and load
        mat2 = bh.load_npz(fname)
        np.testing.assert_array_equal(mat.data, mat2.data)
        assert mat2.diag_num == mat.diag_num
        assert np.all(mat2.shape == mat.shape)


def test_save_and_load(example_band_matrix):
    with tempfile.TemporaryDirectory() as tmpdir:
        mat = example_band_matrix[0]
        fname = os.path.join(tmpdir, "example_bandhic.npz")
        bh.save_npz(fname, mat)
        mat2 = bh.load_npz(fname)
        bh.assert_band_matrix_equal(mat, mat2)


@pytest.mark.parametrize(
    "ufunc",
    [
        np.negative,
        np.positive,
        np.absolute,
        np.sqrt,
        np.square,
        np.exp,
        np.exp2,
        np.expm1,
        np.log,
        np.log2,
        np.log10,
        np.log1p,
        np.sin,
        np.cos,
        np.tan,
        np.arcsin,
        np.arccos,
        np.arctan,
        np.sinh,
        np.cosh,
        np.tanh,
        np.arcsinh,
        np.arccosh,
        np.arctanh,
        np.floor,
        np.ceil,
        np.rint,
        np.trunc,
        np.sign,
        np.degrees,
        np.radians,
        np.deg2rad,
        np.rad2deg,
        np.invert,
        np.reciprocal,
        np.cbrt,
    ],
)
def test_unary_ufuncs(example_band_matrix, ufunc):
    mat, expected_array = example_band_matrix
    # Ensure nonnegative values for log/arcsin etc. to avoid domain errors
    mat.data[:] = np.clip(mat.data, 0.01, None)
    result = ufunc(mat)
    assert isinstance(result, band_hic_matrix)
    assert result.shape == expected_array.shape
    np.testing.assert_array_equal(result.todense(), ufunc(expected_array))


@pytest.mark.parametrize(
    "ufunc",
    [
        np.add,
        np.subtract,
        np.multiply,
        np.true_divide,
        np.floor_divide,
        np.divide,
        np.maximum,
        np.minimum,
        np.power,
        np.mod,
        np.fmod,
        np.remainder,
        np.heaviside,
        np.arctan2,
        np.hypot,
        np.bitwise_and,
        np.bitwise_or,
        np.bitwise_xor,
        np.left_shift,
        np.right_shift,
        np.equal,
        np.not_equal,
        np.greater,
        np.less,
        np.greater_equal,
        np.less_equal,
        np.logical_and,
        np.logical_or,
        np.logical_xor,
        np.gcd,
        np.lcm,
        np.logaddexp,
        np.logaddexp2,
    ],
)
def test_binary_ufuncs(example_band_matrix, ufunc):
    mat, mat_dense = example_band_matrix
    other = mat.copy()
    other_expect = mat_dense.copy()
    result = ufunc(mat, other)
    result_dense = ufunc(mat_dense, other_expect)
    assert isinstance(result, band_hic_matrix)
    assert result.shape == mat_dense.shape
    np.testing.assert_array_equal(result.todense(), result_dense)


# Test operator overloads for basic arithmetic
@pytest.mark.parametrize(
    "op, ufunc",
    [
        (operator.add, np.add),
        (operator.sub, np.subtract),
        (operator.mul, np.multiply),
        (operator.truediv, np.divide),
        (operator.floordiv, np.floor_divide),
        (operator.mod, np.mod),
        (operator.pow, np.power),
        (operator.neg, np.negative),
        (operator.pos, np.positive),
        (operator.abs, np.absolute),
        (operator.and_, np.bitwise_and),
        (operator.or_, np.bitwise_or),
        (operator.xor, np.bitwise_xor),
        (operator.lshift, np.left_shift),
        (operator.rshift, np.right_shift),
        (operator.eq, np.equal),
        (operator.ne, np.not_equal),
        (operator.lt, np.less),
        (operator.le, np.less_equal),
        (operator.gt, np.greater),
        (operator.ge, np.greater_equal),
    ],
)
def test_operator_overloads(example_band_matrix, op, ufunc):
    mat, mat_dense = example_band_matrix
    other = mat.copy()
    # Ensure no division by zero for truediv
    # other.data[:] = np.where(mat.data == 0, 1, mat.data)
    if ufunc.nin == 1:
        # Unary operator
        result = op(mat)
        expected = ufunc(mat_dense)
    else:
        result = op(mat, other)
        expected = ufunc(mat_dense, other.todense())
    # Check type, shape, and values
    assert isinstance(result, band_hic_matrix)
    assert result.shape == expected.shape
    np.testing.assert_array_equal(result.todense(), expected)


# def test_convolve2d(example_band_matrix):
#     mat, expected_array = example_band_matrix
#     kernel = np.ones((3, 3)) / 9
#     conv = bh.convolve2d(mat, kernel)
#     assert isinstance(conv, band_hic_matrix)
#     assert conv.shape == mat.shape
#     # Compare with dense convolution
#     # from scipy.signal import convolve2d as dense_convolve2d
#     # expected_conv = dense_convolve2d(expected_array, kernel, mode='same', boundary='symm')
#     # np.testing.assert_array_almost_equal(conv.todense(), expected_conv)
