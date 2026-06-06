"""
Module: sort_search.py
We are doing data manipulation here by using numpy. We are using sorting and searching
algorithms here. We have included functions like stable_sort, multikey_sort.topk function
is being used here to do partial sorting. binary_search function is being used to search
in sorted array.
"""
import numpy as np


def stable_sort(ar, axis=-1):
    
    ar = np.asarray(ar)

    if axis is None:
        return np.sort(ar.reshape(-1), kind="stable")

    return np.sort(ar, axis=axis, kind="stable")


def multikey_sort(ar, key):
    
    ar = np.asarray(ar)

    if ar.ndim != 2:
        raise ValueError("a 2D array is expected by the multikey_sort")

    if key is None or len(key) == 0:
        raise ValueError("at least a column index should be contained.")

    n_cols = ar.shape[1]

    for col in key:
        if not isinstance(col, (int, np.integer)):
            raise ValueError("all keys should be integer column indice.")
        if col < 0 or col >= n_cols:
            raise IndexError(
                f"Column index {col} is out of bounds for array with {n_cols} columns."
            )

    idx = np.lexsort([ar[:, col] for col in reversed(key)])
    return ar[idx]


def topk(values, k, largest=True, return_indices=True, sorted=True):
    
    values = np.asarray(values)

    if values.ndim != 1:
        raise ValueError("1D array is expected by topk")
    if values.size == 0:
        raise ValueError("It is not possible to operate on empty array")
    if not isinstance(k, (int, np.integer)):
        raise ValueError("k must be an integer.")
    if k < 1 or k > values.size:
        raise ValueError(f"k must satisfy 1 <= k <= {values.size}.")

    if k == values.size:
        idx = np.arange(values.size)
    else:
        if largest:
            idx = np.argpartition(values, -k)[-k:]
        else:
            idx = np.argpartition(values, k - 1)[:k]

    selected_values = values[idx]

    if sorted:
        if largest:
            order = np.argsort(-selected_values, kind="stable")
        else:
            order = np.argsort(selected_values, kind="stable")

        idx = idx[order]
        selected_values = selected_values[order]

    if return_indices:
        return selected_values, idx

    return selected_values


def quickselect(ar, k):
    
    ar = np.asarray(ar)

    if ar.ndim != 1:
        raise ValueError("1D array is expected.")
    if ar.size == 0:
        raise ValueError("cannnot be operated on empty array.")
    if not isinstance(k, (int, np.integer)):
        raise ValueError("k should be integer.")
    if k < 0 or k >= ar.size:
        raise ValueError(f"k should satisfy 0 <= k < {ar.size}.")

    w = ar.copy()
    lt = 0
    rt = w.size - 1

    while True:
        if lt == rt:
            return w[lt]

        pivot_index = (lt + rt) // 2
        pivot_value = w[pivot_index]

        w[pivot_index], w[rt] = w[rt], w[pivot_index]

        store_index = lt
        for i in range(lt, rt):
            if w[i] < pivot_value:
                w[store_index], w[i] = w[i], w[store_index]
                store_index += 1

        w[rt], w[store_index] = w[store_index], w[rt]

        if k == store_index:
            return w[k]
        elif k < store_index:
            rt = store_index - 1
        else:
            lt = store_index + 1


def binary_search(sorted_array, x):
   
    sorted_array = np.asarray(sorted_array)

    if sorted_array.ndim != 1:
        raise ValueError("1D array is expected")

    i = int(np.searchsorted(sorted_array, x, side="left"))
    exists = i < sorted_array.size and sorted_array[i] == x

    return i, bool(exists)


if __name__ == "__main__":
    print(stable_sort(np.array([5, 1, 3, 3])))

    Y = np.array([
        [3, 5],
        [2, 9],
        [2, 8],
        [1, 4]
    ])

    print(multikey_sort(Y, [0, 1]))

    Z = np.array([
        [9, 5],
        [2, 9],
        [2, 8],
        [1, 4]
    ])
    print(multikey_sort(Z, [0, 1]))


    print(topk(np.array([7, 3, 9, 8, 5]), 3, largest=True, return_indices=False))
    print(topk(np.array([7, 3, 9, 8, 5]), 2, largest=True, return_indices=False))
    print(topk(np.array([7, 6, 9, 8, 4]), 4, largest=True, return_indices=False))
    



    
    print(binary_search(np.array([1, 3, 5, 7]), 5))

    print(binary_search(np.array([1, 3, 5, 7]), 3))
    
    print(quickselect(np.array([7, 2, 9, 1, 5]), 2))
    
    print(quickselect(np.array([6, 3, 8, 1, 6]), 2))
    print(quickselect(np.array([4, 3, 8, 1, 7]), 2))
