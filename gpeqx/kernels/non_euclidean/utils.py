# 

from jaxtyping import (
    Float,
    Int,
)

from gpepx.typing import Array


def jax_gather_nd(
    params: Float[Array, " N *rest"], indices: Int[Array, " M 1"]
) -> Float[Array, " M *rest"]:
    r"""Slice a `params` array at a set of `indices`.

    This is a reimplementation of TensorFlow's `gather_nd` function:
    [link](https://www.tensorflow.org/api_docs/python/tf/gather_nd)

    Args:
        params: an arbitrary array with leading axes of length $N$ upon
            which we shall slice.
        indices: an integer array of length $M$ with values in the range
            $[0, N)$ whose value at index $i$ will be used to slice `params` at
            index $i$.

    Returns:
        An arbitrary array with leading axes of length $M$.
    """
    tuple_indices = tuple(indices[..., i] for i in range(indices.shape[-1]))
    return params[tuple_indices]
