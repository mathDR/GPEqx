# 

import beartype.typing as tp
from jax import vmap
import jax.numpy as jnp
from jaxtyping import Float

import gpepx  # noqa: F401
from gpepx.kernels.computations import AbstractKernelComputation
from gpepx.typing import Array

Kernel = tp.TypeVar("Kernel", bound="gpepx.kernels.base.AbstractKernel")  # noqa: F821


class DiagonalKernelComputation(AbstractKernelComputation):
    r"""Diagonal kernel computation class. Operations with the kernel assume
    a diagonal Gram matrix.
    """

    def gram(self, kernel: Kernel, x: Float[Array, "N D"]) -> Float[Array, "N N"]:
        return jnp.diag(vmap(lambda x: kernel(x, x))(x))

    def _cross_covariance(
        self, kernel: Kernel, x: Float[Array, "N D"], y: Float[Array, "M D"]
    ) -> Float[Array, "N M"]:
        # TODO: This is currently a dense implementation.
        # We should implement a sparse LinearOperator for non-square cross-covariance matrices.
        cross_cov = vmap(lambda x: vmap(lambda y: kernel(x, y))(y))(x)
        return cross_cov
