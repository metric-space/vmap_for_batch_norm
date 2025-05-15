from typing import Tuple, List, Dict, Any, Union

import jax
import jax.random as jr
import jax.numpy as jnp
from .datasets import mnist
from jaxtyping import PyTree, Float32, Array, Int32, PRNGKeyArray
from collections.abc import Callable
import numpy as np
import grain
import operator


# ======== Types ===============

ParamType = Float32[Array, "w h"]
NNParams = List[Dict[str, ParamType]]
BatchParams = List[ParamType]
ParamsType = Union[NNParams, Tuple[NNParams, BatchParams]]

# =============================


# m rows and n_cols
def random_layer(key: PRNGKeyArray, m: int, n: int, affine=True) -> ParamType:
    w_key, b_key = jr.split(key, 2)

    layer = dict(weight=jr.normal(w_key, (m, n)))

    if affine:
        layer["bias"] = jr.normal(b_key, (n,))

    return layer


def init_nn_params(key: PRNGKeyArray, arch: List[Tuple[int, int]]) ->  NNParams:
    nn = []

    keys = jr.split(key, len(arch))

    for i, layer in enumerate(arch):

        init_layer = random_layer(
            keys[i], layer[0], layer[1], affine=True if i != len(arch) - 1 else False
        )

        nn.append(init_layer)

    return nn


def named_grad_norms(grads) -> str:
    """

    Tool to list and pretty print gradient flow through the model

    """
    flat = jax.tree_util.tree_flatten_with_path(grads)[0]
    return {
        ".".join(str(k) for k in path): jnp.sqrt(jnp.sum(leaf**2))
        for path, leaf in flat
        if leaf is not None
    }


def mnist_dataloader(directory:str, batch_size: int=2, seed: int=14321) -> Tuple[grain.DataLoader, grain.DataLoader]:
    train_data, train_labels, test_data, test_labels = mnist(directory)
    trainloader = grain.load(
        list(zip(train_data, train_labels)),
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )
    testloader = grain.load(
        list(zip(test_data, test_labels)),
        batch_size=batch_size,
        num_epochs=1,
        shuffle=True,
        seed=seed,
    )
    return trainloader, testloader


def forward_pass(nn: List[Float32[Array, "y x"]], image_vector: Float32[Array, "x"]) -> Float32[Array, "x"]:

    assert image_vector.ndim == 1

    output = image_vector

    for i, lt in enumerate(nn):
        w = lt["weight"]

        output = output @ w
        if lt.get("bias", None) is not None:
            output += lt["bias"]

        if i < len(nn) - 1:
            output = jax.nn.sigmoid(output)

    assert output.ndim == 1  #

    return output


def cross_entropy_loss(params: NNParams , x: Float32[Array, "batch x"] , labels: Int32[Array, "batch z"]) -> Float32:

    logits = jax.vmap(forward_pass, in_axes=(None, 0))(params, x)
    # one-hot, this is batched
    logits = jax.nn.log_softmax(logits, axis=-1)
    loss = jax.vmap(operator.getitem, in_axes=(0, 0))(logits, labels)
    return -loss.mean()


@jax.jit
def update(params, x, y):
    step_size = 0.01
    vals, grads = jax.value_and_grad(cross_entropy_loss)(params, x, y)

    updated_params = [
        {
            "weight": params["weight"] - step_size * dparams["weight"],
            "bias": (
                None
                if params.get("bias", None) == None
                else (params["bias"] - step_size * dparams["bias"])
            ),
        }
        for params, dparams in zip(params, grads)
    ]

    return vals, grads, updated_params


# TODO: check if typing makes sense
def evaluate_model(
    test_loader,
    forward_pass: Callable[
        [ParamType, Float32[Array, "batch flattened_images"]],
        Float32[Array, "batch predictions"],
    ],
    *args,
) -> Float32:

    total_correct = 0
    total = 0

    for batch_x, batch_y in test_loader:

        preds = forward_pass(*args, batch_x)
        predicted_class = jnp.argmax(preds, axis=-1)

        total_correct += (predicted_class == jnp.array(batch_y)).sum()
        total += batch_x.shape[0]

    acc = total_correct / total

    print(f"Evaluation accuracy: {acc:.4f}")
    return acc
