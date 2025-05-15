import jax
import jax.lax as l
import jax.numpy as jnp

import itertools
import operator


# I believe activations are of shape = hidden layer width
def activations_batch_normalization(activations, axis, eta=1e-6):
    mean = l.pmean(activations, axis_name=axis)
    sq_mean = jax.lax.pmean(activations**2, axis_name="batch")
    var = sq_mean - mean**2

    return ((activations - mean) / jnp.sqrt(var + eta)), mean, var


def forward_pass(nn, batch_norm_params, axis, image_vector):

    assert image_vector.ndim == 1

    output = image_vector

    means, variances = [], []

    for i, (lt, bn) in enumerate(itertools.zip_longest(nn, batch_norm_params)):
        w = lt["weight"]

        output = output @ w

        if i < len(nn) - 1:
            output += lt["bias"]

            x_i, mean, var = activations_batch_normalization(output, axis)

            means.append(mean)
            variances.append(var)

        if bn is not None:
            output = x_i * bn[:, 0] + bn[:, 1]

        if i < len(nn) - 1:
            output = jax.nn.sigmoid(output)

    assert output.ndim == 1

    # the means, variances collection effort is duplicated across batches

    return output, means, variances


def cross_entropy_loss(params, x, labels):

    axis = "batch"

    params, batch_norm_params = params
    logits, means, variances = jax.vmap(
        forward_pass,
        in_axes=(None, None, None, 0),
        out_axes=(0, None, None),
        axis_name=axis,
    )(params, batch_norm_params, axis, x)

    logits = jax.nn.log_softmax(logits, axis=-1)
    loss = jax.vmap(operator.getitem, in_axes=(0, 0))(logits, labels)
    return -loss.mean(), (means, variances)
