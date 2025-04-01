# 

import beartype.typing as tp
import jax.numpy as jnp
from jaxtyping import (
    Float,
    Num,
)

import gpepx  # noqa: F401
from gpepx.kernels.computations.base import AbstractKernelComputation
from gpepx.typing import Array

Kernel = tp.TypeVar(
    "Kernel",
    bound="gpepx.kernels.non_euclidean.graph.GraphKernel",  # noqa: F821
)


class EigenKernelComputation(AbstractKernelComputation):
    r"""Eigen kernel computation class. Kernels who operate on an
    eigen-decomposed structure should use this computation object.
    """

    def _cross_covariance(
        self, kernel: Kernel, x: Num[Array, "N D"], y: Num[Array, "M D"]
    ) -> Float[Array, "N M"]:
        # Transform the eigenvalues of the graph Laplacian according to the
        # RBF kernel's SPDE form.
        S = jnp.power(
            kernel.eigenvalues.value
            + 2
            * kernel.smoothness.value
            / kernel.lengthscale.value
            / kernel.lengthscale.value,
            -kernel.smoothness.value,
        )
        S = jnp.multiply(S, kernel.num_vertex / jnp.sum(S))
        # Scale the transform eigenvalues by the kernel variance
        S = jnp.multiply(S, kernel.variance.value)
        return kernel(x, y, S=S)
