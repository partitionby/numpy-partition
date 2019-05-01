import numpy as np
from functools import partial
from partition.window import row_number, top, avg


def partition(data, partition_by_col_indexes, value_col_indexes, value_ordering):
    """ Sorts and splits data by partition_by_col_indexes.

    Returns data ordering, sorted dataset, partition boundaries, split indexes, partitions.

    TODO: value_* arguments do not belong here

    >>> data = np.array([[1,1,3], [2,2,3], [1,1,4]], dtype=np.float32)
    >>> partition_by_col_indexes = (0, 1)
    >>> value_col_indexes = (2,)
    >>> value_ordering = (-1,)  # descending order
    >>> result = partition(data=data, partition_by_col_indexes=partition_by_col_indexes, value_col_indexes=value_col_indexes, value_ordering=value_ordering)
    >>> data_ordering, data_sorted, partition_boundaries, split_indexes, partitions = result
    >>> data_ordering
    array([2, 0, 1])
    >>> data_sorted
    array([[1., 1., 4.],
           [1., 1., 3.],
           [2., 2., 3.]], dtype=float32)
    >>> partition_boundaries
    array([ True, False,  True])
    >>> split_indexes
    array([0, 2])
    >>> partitions
    [array([[1., 1., 4.],
           [1., 1., 3.]], dtype=float32), array([[2., 2., 3.]], dtype=float32)]

    """
    # Order
    values = np.hstack(list(map(lambda x, y: x * data[:, (y,)], value_ordering, value_col_indexes)))
    partition_by = data[:, list(partition_by_col_indexes)[::-1]]
    sort_columns = np.hstack((values, partition_by))
    data_ordering = np.lexsort(sort_columns.T, axis=0)
    data_sorted = data[data_ordering]
    # Partition
    diffs = np.vstack(
        (
            np.repeat(True, len(partition_by_col_indexes)),
            np.diff(data_sorted[:, partition_by_col_indexes], axis=0) != 0
        )
    )
    partition_boundaries = np.logical_or.reduce(diffs[:, [i for i, _ in enumerate(partition_by_col_indexes)]], axis=1)
    (split_indexes,) = np.where(partition_boundaries == True)
    partitions = np.vsplit(data_sorted, split_indexes[split_indexes > 0])
    return data_ordering, data_sorted, partition_boundaries, split_indexes, partitions


def partition_number(data, partition_by_col_indexes, value_col_indexes, value_ordering):
    """ Returns partition number for each row.

    Numbering starts from 0.

    TODO: value_* arguments do not belong here but are still required by partition()

    >>> data = np.array([[1,1,3], [2,2,3], [1,1,4]], dtype=np.float32)
    >>> partition_by_col_indexes = (0, 1)
    >>> value_col_indexes = (2,)
    >>> value_ordering = (-1,)  # descending order
    >>> partition_number(data=data, partition_by_col_indexes=partition_by_col_indexes, value_col_indexes=value_col_indexes, value_ordering=value_ordering)
    array([1, 2, 1])

    """
    data_ordering, _, partition_boundaries, _, _ = partition(data, partition_by_col_indexes, value_col_indexes, value_ordering)
    return np.cumsum(partition_boundaries)[np.argsort(data_ordering)]


def apply_over_partition(data, partition_by_col_indexes, value_col_indexes, value_ordering, f, f_kwargs):
    """ Applies a provided window function to each partition of the dataset.

    Returns a numpy array with the concatenated results for all partitions.


    >>> data = np.array([[1,1,3], [2,2,3], [1,1,4]], dtype=np.float32)
    >>> partition_by_col_indexes = (0, 1)
    >>> value_col_indexes = (2,)
    >>> value_ordering = (-1,)  # descending order
    >>> f = avg
    >>> f_kwargs = dict(vcol=2, top_n=2)
    >>> apply_over_partition(data=data, partition_by_col_indexes=partition_by_col_indexes, value_col_indexes=value_col_indexes, value_ordering=value_ordering, f=f, f_kwargs=f_kwargs)
    array([3.5, 3. , 3.5])

    >>> f = avg
    >>> f_kwargs = dict(vcol=2, top_n=1)
    >>> apply_over_partition(data=data, partition_by_col_indexes=partition_by_col_indexes, value_col_indexes=value_col_indexes, value_ordering=value_ordering, f=f, f_kwargs=f_kwargs)
    array([4., 3., 4.])

    >>> f = row_number
    >>> f_kwargs = dict()
    >>> apply_over_partition(data=data, partition_by_col_indexes=partition_by_col_indexes, value_col_indexes=value_col_indexes, value_ordering=value_ordering, f=f, f_kwargs=f_kwargs)
    array([1, 0, 0])

    """
    data_ordering, _, _, _, partitions = partition(data, partition_by_col_indexes, value_col_indexes, value_ordering)
    f_results = np.hstack(list(map(partial(f, **f_kwargs), partitions)))
    # Restore original ordering
    result = f_results[np.argsort(data_ordering)]
    return result

