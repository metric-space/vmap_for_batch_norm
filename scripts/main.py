import jax
import jax.numpy as jnp
import jax.random as jr

# import wandb
import operator
import itertools

from typing import Tuple, List
import pprint as pp
from jaxtyping import Array, Float

import common
import batch_norm
import batch_norm_vanilla

from dataclasses import dataclass, field

import matplotlib.pyplot as plt


@dataclass
class Config:
    batch_size: int = 100
    arch: List[Tuple[int, int]] = field(
        default_factory=lambda: [(784, 100), (100, 100), (100, 100), (100, 10)]
    )
    steps: int = 2000
    seed: int = 1232


def init_params(key, arch: List[Tuple[int, int]]):

    n_key, b_key = jr.split(key, 2)

    return common.init_nn_params(
        n_key, arch
    ), batch_norm_vanilla.init_batch_norm_parameters(b_key, arch)


# output type is List of jax.arrays
def calculate_running_stats(running_mean, running_var):

    running_mean = [jnp.stack(layer).mean(axis=0) for layer in zip(*running_mean)]
    running_var = [jnp.stack(layer).mean(axis=0) for layer in zip(*running_var)]

    return (running_mean, running_var)


if __name__ == "__main__":

    results = {"baseline": [], "vanilla_batch_norm": [], "batch_norm": []}

    config = Config()

    trainloader, test_loader = common.mnist_dataloader(
        directory="./MNIST", batch_size=config.batch_size
    )

    key = jr.PRNGKey(config.seed)

    print("====== Training and evaluating without batch norm ========")

    params, _ = init_params(key, config.arch)

    for step, (batch_x, batch_y) in zip(range(config.steps), trainloader):

        (
            loss,
            grads,
            params,
        ) = common.update(params, batch_x, batch_y)

        if step % 5 == 0:
            print(f"Loss :{loss} for Step: {step}")
            f = jax.vmap(common.forward_pass, in_axes=(None, 0))
            accuracy = common.evaluate_model(test_loader, f, params)

            results["baseline"].append((step, accuracy))

    print("====== Training and evaluating with vanilla batch norm ========")

    trainloader, test_loader = common.mnist_dataloader(
        directory="./MNIST", batch_size=config.batch_size
    )

    params, batch_norm_params = init_params(key, config.arch)

    means, variances = [], []

    for step, (batch_x, batch_y) in zip(range(config.steps), trainloader):

        loss, grads, params, batch_norm_params, means_, variances_ = (
            batch_norm_vanilla.update(
                batch_norm_vanilla.cross_entropy_loss,
                params,
                batch_norm_params,
                batch_x,
                batch_y,
            )
        )

        means.append(means_)
        variances.append(variances_)

        if step % 5 == 0:
            print(f"Loss :{loss} for Step: {step}")

            running_mean, running_var = calculate_running_stats(means, variances)

            f = jax.vmap(
                batch_norm_vanilla.forward_pass_inference, in_axes=(None, None, None, 0)
            )
            accuracy = common.evaluate_model(
                test_loader, f, params, batch_norm_params, (running_mean, running_var)
            )

            results["vanilla_batch_norm"].append((step, accuracy))

    print("====== Training and evaluating with batch norm ========")

    params, batch_norm_params = init_params(key, config.arch)

    means, variances = [], []

    for step, (batch_x, batch_y) in zip(range(config.steps), trainloader):

        loss, grads, params, batch_norm_params, means_, variances_ = (
            batch_norm_vanilla.update(
                batch_norm.cross_entropy_loss,
                params,
                batch_norm_params,
                batch_x,
                batch_y,
            )
        )

        means.append(means_)
        variances.append(variances_)

        if step % 5 == 0:
            print(f"Loss :{loss} for Step: {step}")

            running_mean, running_var = calculate_running_stats(means, variances)

            f = jax.vmap(
                batch_norm_vanilla.forward_pass_inference, in_axes=(None, None, None, 0)
            )
            accuracy = common.evaluate_model(
                test_loader, f, params, batch_norm_params, (running_mean, running_var)
            )

            results["batch_norm"].append((step, accuracy))

    # ============ plotting the results =======================

    fig, ax = plt.subplots(figsize=(15, 15))
    x, y = zip(*results["baseline"])
    plt.plot(x, y, color="red")
    x, y = zip(*results["vanilla_batch_norm"])
    plt.plot(x, y, color="green")
    x, y = zip(*results["batch_norm"])
    plt.plot(x, y, color="blue")

    plt.savefig("output.png")

    # print(batch_y)
