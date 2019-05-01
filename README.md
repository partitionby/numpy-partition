# Split numpy arrays into partitions by one or multiple columns and apply window function to each partition.


This module tries to replicate `select window_function() over (partition by ... order by ...) ...` functionality, commonly found in SQL databases.

The following window functions are available out of the box: `row_number()`, `top()`, `avg()`. 


## Usage examples:

```
    >>> from partition import apply_over_partition
    >>> from partition.window import row_number, top, avg

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

    >>> f = top
    >>> f_kwargs = dict(n=1)
    >>> apply_over_partition(data=data, partition_by_col_indexes=partition_by_col_indexes, value_col_indexes=value_col_indexes, value_ordering=value_ordering, f=f, f_kwargs=f_kwargs)
    array([False,  True,  True])

```

