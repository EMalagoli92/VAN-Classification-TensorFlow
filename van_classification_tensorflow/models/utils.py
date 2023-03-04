import collections.abc
from itertools import repeat
from typing import Any, Callable, Iterable

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


def _ntuple(n: int) -> Callable[[Any], collections.abc.Iterable[Any]]:
    """
    Parameters
    ----------
    n : int
        The length of the desired tuple.

    Returns
    -------
    Callable[[Any], collections.abc.Iterable[Any]]
        A function such that:
            - if the input is an iterable returns the input;
            - if the input is not an iterable returns a tuple of n elements, all equal
              to the input.
    """

    def parse(x: Any) -> collections.abc.Iterable[Any]:
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse
