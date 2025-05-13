import jax
import jax.numpy as jnp
import jax.random as jr
# import wandb
import orbax
import operator
import itertools

import torch
import torchvision
import torchvision.transforms as transforms

from typing import Tuple, List
import pprint as pp
from jaxtyping import Array, Float 

import common
import batch_norm_vanilla

from dataclasses import dataclass


@datclass
data Config:
    batch_size: int = 100
    arch: List[Tuple[int,int]] = [(784,100), (100,100), (100,100), (100,10)]
    steps: int = 2000
    seed: int = 1232


def init_params(key, arch: List[Tuple[int,int]]):

    n_key, b_key = jr.split(key, 2)

    return common.init_nn_params(n_key, arch), batch_norm_vanilla.init_batch_norm_parameters(b_key, arch)



# output type is List of jax.arrays
def calculate_running_stats(running_mean, running_var):

     running_mean = [jnp.stack(layer).mean(axis=0) for layer in zip(*running_mean)]
     running_var = [jnp.stack(layer).mean(axis=0) for layer in zip(*running_var)]

     return (running_mean, running_var)


if __name__ == '__main__':

    results = {"baseline":[], "vanilla_batch_norm": [], "batch_norm": []}

    config = Config()

    trainloader = common.dataset(train=True, batch_size=config.batch_size)

    key = jr.PRNGKey(config.seed)

    print("====== Training and evaluating without batch norm ========")

    params, _ = init_params(key, config.arch)

    for step, (batch_x, batch_y) in zip(range(config.steps), common.inf_dataloader(trainloader)):

        batch_x = batch_x.numpy().reshape(batch_x.shape[0],-1)
        batch_y = batch_y.numpy()

        loss, grads, params, = common.update(params, batch_x, batch_y)

        if step % 100 == 0 :
            print(f"Loss :{loss} for Step: {step}")
            accuracy = common.evaluate_model(params, batch_norm_parameters)

            result["baseline"].append((step, accuracy))


    print("====== Training and evaluating with vanilla batch norm ========")

    params, batch_norm_params = init_params(key, config.arch)

    means, variances = [], []

    for step, (batch_x, batch_y) in zip(range(config.steps), common.inf_dataloader(trainloader)):

        batch_x = batch_x.numpy().reshape(batch_x.shape[0],-1)
        batch_y = batch_y.numpy()

        loss, grads, params, means_, variances_ = batch_norm_vanilla.update(batch_norm_vanilla.cross_entropy, params, batch_x, batch_y)

        means.append(means_)
        variances.append(means_)

        if step % 100 == 0 :
            print(f"Loss :{loss} for Step: {step}")

            running_mean, running_var = calculate_running_stats(means_, vars_)

            f = jax.vmap(batch_norm_vanilla.forward_pass_inference, in_axes=(None, None, None, 0))
            accuracy = common.evaluate_model(f, batch_y, nn=params, batch_norm_params=batch_norm_params, running_stats=(running_mean, running_var), image_vec=batch_x)

            result["vanilla_batch_norm"].append((step, accuracy))


    print("====== Training and evaluating with batch norm ========")

    params, batch_norm_params = init_params(key, config.arch)

    means, variances = [], []

    for step, (batch_x, batch_y) in zip(range(config.steps), common.inf_dataloader(trainloader)):

        batch_x = batch_x.numpy().reshape(batch_x.shape[0],-1)
        batch_y = batch_y.numpy()

        loss, grads, params, means_, variances_ = batch_norm.update(batch_norm.cross_entropy, params, batch_x, batch_y)

        means.append(means_)
        variances.append(means_)

        if step % 100 == 0 :
            print(f"Loss :{loss} for Step: {step}")

            running_mean, running_var = calculate_running_stats(means_, vars_)

            f = jax.vmap(batch_norm_vanilla.forward_pass_inference, in_axes=(None, None, None, 0))
            accuracy = common.evaluate_model(f, batch_y, nn=params, batch_norm_params=batch_norm_params, running_stats=(running_mean, running_var), image_vec=batch_x)

            result["vanilla_batch_norm"].append((step, accuracy))


    #print(batch_y)
