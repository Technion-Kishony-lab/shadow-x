import numpy as np
import pytest


def group_argmax_sorted(indices, vals):
    """
    For each group in `indices`, find the index of the first occurrence of the maximum value in `vals`.
    """
    # Identify where groups start and end
    unique_idx, start = np.unique(indices, return_index=True)

    # Compute max per group (vectorized)
    max_vals = np.maximum.reduceat(vals, start)

    # Build mask of where vals == group max (vectorized broadcasting)
    # This gives a boolean array the same length as vals, True where each val == its group's max
    group_max_for_each = np.repeat(max_vals, np.diff(np.r_[start, len(vals)]))
    mask = vals == group_max_for_each

    # Keep only first occurrence within each group
    first_in_group = np.r_[True, indices[1:] != indices[:-1]]
    new_group = np.cumsum(first_in_group) - 1
    # The first True in each group (like argmax)
    _, first_true_idx = np.unique(new_group[mask], return_index=True)
    argmax_idx = np.flatnonzero(mask)[first_true_idx]

    return unique_idx, argmax_idx, max_vals


def cyclic_true_center_last_axis(a):
    """
    Find the center of the single longest true run in the last axis of a boolean array.
    Assumes one series of True values, possibly cyclic.
    """
    a = np.asarray(a, dtype=bool)
    *prefix_shape, L = a.shape
    flat = a.reshape(-1, L)
    if L == 0:
        return np.full(prefix_shape, -1)

    # Double the array to handle cyclic nature
    doubled = np.concatenate([flat, flat], axis=1)

    # Assume only one true sequence
    int_doubled = doubled.astype(int)
    int_doubled = np.pad(int_doubled, ((0, 0), (1, 1)), mode='constant', constant_values=0)
    diff = np.diff(int_doubled, axis=1)
    start_idx = np.flatnonzero(diff == 1)
    end_idx = np.flatnonzero(diff == -1)
    lengths = end_idx - start_idx
    row_idx_for_each_length = start_idx // (2 * L + 1)
    start_idx = start_idx % (2 * L + 1)
    end_idx = end_idx % (2 * L + 1)
    middle = (start_idx + end_idx - 1) // 2 % L
    middle[(start_idx == 0) & (end_idx == 2 * L)] = -1  # full true cycle
    unique_rows, argmax_idx, _ = group_argmax_sorted(row_idx_for_each_length, lengths)
    result = np.full(flat.shape[0], -1, dtype=int)
    result[unique_rows] = middle[argmax_idx]
    return result.reshape(prefix_shape)


@pytest.mark.parametrize("arr, expected", [
    (np.array([0, 0, 1, 1, 1, 0, 0, 0]), 3),
    (np.array([[1, 1, 1, 0, 0], [1, 1, 0, 0, 1], [1, 0, 0, 1, 1]]), np.array([1, 0, 4])),
    (np.array([[0, 0, 1, 1, 1, 0, 0, 0], [0, 0, 1, 1, 1, 0, 0, 0], [0, 0, 1, 1, 1, 0, 0, 0]]), np.array([3, 3, 3])),
    (np.array([[1, 1, 0, 1, 0, 0, 1], [0, 0, 1, 1, 1, 0, 0]]), np.array([0, 3])),
    (np.array([[0, 0, 0, 0], [0, 1, 0, 0]]), np.array([-1, 1])),
    (np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]), np.array([-1, -1])),  # a cycle has no center
])
def test_cyclic_true_center_last_axis(arr, expected):
    result = cyclic_true_center_last_axis(arr)
    if isinstance(expected, np.ndarray):
        assert np.all(result == expected)
    else:
        assert result == expected
