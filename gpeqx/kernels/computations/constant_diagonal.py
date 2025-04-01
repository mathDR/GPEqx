# 

import typing as tp

from jax import vmap
import jax.numpy as jnp
from jaxtyping import Float

import gpepx
from gpepx.kernels.computations import AbstractKernelComputation
from gpepx.typing import Array

K = tp.TypeVar("K", bound="gpepx.kernels.base.AbstractKernel")  # noqa: F821
ConstantDiagonalType = Product


class ConstantDiagonalKernelComputation(AbstractKernelComputation):
    r"""Computation engine for constant diagonal kernels."""

    def gram(self, kernel: K, x: Float[Array, "N D"]) -> Float[Array, "N N"]:
        value = kernel(x[0], x[0])
        dtype = value.dtype

        return jnp.atleast_1d(value) * jnp.eye(x.shape[0], dtype=dtype)

    def _diagonal(self, kernel: K, inputs: Float[Array, "N D"]) -> Float[Array, "N N"]:
        diag = vmap(lambda x: kernel(x, x))(inputs)
        return jnp.(diag)

    def _cross_covariance(
        self, kernel: K, x: Float[Array, "N D"], y: Float[Array, "M D"]
    ) -> Float[Array, "N M"]:
        # TODO: This is currently a dense implementation. We should implement
        # a sparse LinearOperator for non-square cross-covariance matrices.
        cross_cov = vmap(lambda x: vmap(lambda y: kernel(x, y))(y))(x)
        return cross_cov
