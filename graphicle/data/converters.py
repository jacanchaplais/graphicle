import numpy as np


# TODO: add reverse from structured to unstructured
def cast_array(in_array: np.ndarray, cast_type) -> np.ndarray:
    cast_type = np.dtype(cast_type)
    if in_array.dtype != cast_type:
        multi_dim = in_array.ndim > 1
        input_col_num = in_array.shape[-1]
        target_col_num = len(cast_type.fields)
        if multi_dim and input_col_num != target_col_num:
            raise ValueError(
                    f'Casting to {cast_type} requires in_array with '
                    + f'{target_col_num} columns, but you passed an '
                    + f'array with shape {in_array.shape}.'
                    )
        cast_array = in_array.astype(cast_type.descr[0][1])
        cast_array = cast_array.view(dtype=cast_type, type=np.ndarray)
        cast_array = cast_array.copy().squeeze()
    else:
        cast_array = in_array
    return cast_array
