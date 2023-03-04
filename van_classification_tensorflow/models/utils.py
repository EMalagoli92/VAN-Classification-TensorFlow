from typing import List, Tuple, Union

import tensorflow as tf


def _to_channel_last(x: tf.Tensor) -> tf.Tensor:
    """
    Parameters
    ----------
    x : tf.Tensor
        Tensor of shape: (B, C, H, W).

    Returns
    -------
    tf.Tensor
        Tensor of shape: (B, H, W, C).
    """
    return tf.transpose(x, perm=[0, 2, 3, 1])


def _to_channel_first(x: tf.Tensor) -> tf.Tensor:
    """
    Parameters
    ----------
    x : tf.Tensor
        Tensor of shape: (B, H, W, C).

    Returns
    -------
    tf.Tensor
        Tensor of shape: (B, C, H, W).
    """
    return tf.transpose(x, perm=[0, 3, 1, 2])


def _imgr2tuple(
    x: Union[int, Tuple[int, int], List[int]]
) -> Union[Tuple[int, int], List[int]]:
    """
    Parameters
    ----------
    x : Union[int, Tuple[int, int], List[int]]
        Image resolution.

    Raises
    ------
    ValueError
        If image resolution is neither an integer nor a tuple/list of 2 integers.

    Returns
    -------
    Union[Tuple[int, int], List[int]]
        Image resolution casted in tuple/list of 2 integers.
    """
    if (
        isinstance(x, (tuple, list))
        and (len(x) == 2)
        and all(isinstance(x_, int) for x_ in x)
    ):
        return x
    elif type(x) == int:
        return (x, x)
    else:
        raise ValueError(
            "Image resolution must be an integer or tuple/list of 2 integers."
        )
