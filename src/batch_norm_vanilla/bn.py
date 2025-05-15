import itertools
import operator

import jax
import jax.random as jr
import jax.numpy as jnp

from typing import List, Tuple

from functools import partial

from jaxtyping import PRNGKeyArray, Float32, Array, Int32
import common


def init_batch_norm_parameters(
    key: PRNGKeyArray, arch: List[Tuple[int, int]]
) -> common.BatchParams:

    shapes = [(layer_shape[1], 2) for layer_shape in arch[:-1]]

    keys = jr.split(key, len(shapes))

    return [jr.normal(key, shape) for key, shape in zip(keys, shapes)]


def activations_batch_normalization(
    activations: Float32[Array, "batch x"], eta: Float32 = 1e-6
) -> Tuple[Float32[Array, "batch x"], Float32[Array, "x"], Float32[Array, "x"]]:

    mean = activations.mean(axis=0)  # (B,K) -> (K)
    var = jnp.var(activations, axis=0, ddof=0)  # (B,K) -> (K,)
    x_i = (activations - mean[None, :]) / jnp.pow(var[None, :] + eta, 0.5)

    return x_i, mean, var


def forward_pass(
    nn: common.NNParams,
    batch_norm_params: common.BatchParams,
    image_vector: Float32[Array, "batch x"],
) -> Tuple[Float32[Array, "batch classes"], Float32[Array, "z"], Float32[Array, "z"]]:

    assert image_vector.ndim == 2

    output = image_vector

    means_ = []
    vars_ = []

    for i, (lt, bn) in enumerate(itertools.zip_longest(nn, batch_norm_params)):
        w = lt["weight"]

        output = output @ w
        if lt.get("bias", None) is not None:
            output += lt["bias"]
        else:
            continue

        output, mean, var = activations_batch_normalization(output)

        means_.append(mean)
        vars_.append(var)

        if bn is not None:
            output = output * bn[:, 0][None, :] + bn[:, 1][None, :]

        if i < len(nn) - 1:
            output = jax.nn.sigmoid(output)

    assert output.ndim == 2

    return output, means_, vars_


def forward_pass_inference(
    nn: common.NNParams,
    batch_norm_params: common.BatchParams,
    running_stats: Tuple[Float32[Array, "z"], Float32[Array, "z"]],
    image_vector: Float32[Array, "x"],
) -> Float32[Array, "classes"]:

    assert image_vector.ndim == 1

    output = image_vector

    for i, (lt, bn, running_mean, running_var) in enumerate(
        itertools.zip_longest(nn, batch_norm_params, running_stats[0], running_stats[1])
    ):

        w = lt["weight"]

        output = output @ w
        if i < len(nn) - 1 and lt.get("bias", None) is not None:
            output += lt["bias"]
        else:
            continue

        if bn is not None:
            x_i = (output - running_mean) / jnp.sqrt(running_var + 1e-6)
            output = x_i * bn[:, 0] + bn[:, 1]

        if i < len(nn) - 1:
            output = jax.nn.sigmoid(output)

    return output


def cross_entropy_loss(params: common.ParamsType, x: Float32[Array, "batch x"] , labels: Int32[Array, "batch classes"]):

    params, batch_norm_params = params

    # logits = jax.vmap(forward_pass, in_axes=(None,0))(params, x)
    logits, means, vars_ = forward_pass(params, batch_norm_params, x)

    # one-hot, this is batched
    logits = jax.nn.log_softmax(logits, axis=-1)
    loss = jax.vmap(operator.getitem, in_axes=(0, 0))(logits, labels)
    return -loss.mean(), (means, vars_)


@partial(jax.jit, static_argnums=0)
def update(cross_entropy_loss, params, batch_norm_params, x, y):
    # TODO: extract step_size
    step_size = 0.01
    (vals, aux), grads = jax.value_and_grad(cross_entropy_loss, has_aux=True)(
        (params, batch_norm_params), x, y
    )

    updated_params = [
        {
            "weight": params["weight"] - step_size * dparams["weight"],
            "bias": (
                None
                if params.get("bias", None) == None
                else (params["bias"] - step_size * dparams["bias"])
            ),
        }
        for params, dparams in zip(params, grads[0])
    ]
    updated_batch_params = [
        (param - step_size * dparam)
        for (param, dparam) in zip(batch_norm_params, grads[1])
    ]

    return vals, grads, updated_params, updated_batch_params, aux[0], aux[1]
