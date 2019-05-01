import numpy as np


def row_number(w, axis=0):
    """ Returns row numbers (starting from 0) over specified axis

    >>> w = np.array([[1,1,3], [1,1,4]], dtype=np.float32)
    >>> row_number(w)
    array([0, 1])

    """
    return np.argsort(np.zeros(w.shape[axis]))


def top(w, n, axis=0):
    """ Returns a filter for top n rows over specified axis

    >>> w = np.array([[1,1,3], [1,1,4]], dtype=np.float32)
    >>> top(w, n=1)
    array([ True, False])

    """
    return row_number(w, axis=axis) < n


def avg(w, vcol, top_n=None):
    """ Returns average for `top_n` items in column with index `vcol` over axis 0.

    If top_n is None, the average calculated over all rows.

    >>> w = np.array([[1,1,4], [1,1,3]], dtype=np.float32)
    >>> avg(w, vcol=2, top_n=2)
    array([3.5, 3.5])

    >>> avg(w, vcol=2, top_n=1)
    array([4., 4.])

    """
    row_count = w.shape[0]
    if top_n is None:
        values = w[:, vcol]
    else:
        values = w[:, vcol][top(w, n=top_n, axis=0)]
    avg_result = np.sum(values) / min(top_n, row_count)
    return np.repeat(avg_result, row_count)

