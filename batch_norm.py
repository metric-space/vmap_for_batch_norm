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


config = dict(
        batch_size=100,
        )


transform = transforms.Compose([ transforms.ToTensor() ])
train_dataset = torchvision.datasets.MNIST("MNIST", train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader( train_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True)


def named_grad_norms(grads):
    flat = jax.tree_util.tree_flatten_with_path(grads)[0]
    return {
        ".".join(str(k) for k in path): jnp.sqrt(jnp.sum(leaf**2))
        for path, leaf in flat if leaf is not None
    }


# m rows and n_cols
def random_layer(key, m, n, affine=True):
    w_key, b_key = jr.split(key,2)

    layer = dict(weight=jr.normal(w_key, (m,n)))

    if affine:
        layer["bias"] = jr.normal(b_key, (n,))

    return layer


def init_batch_norm_parameters(key, arch: List[Tuple[int,int]]):

     batch_norm_parameters = []

     shapes = [(layer_shape[1],2) for layer_shape in arch[:-1]]

     keys = jr.split(key, len(shapes))

     return [jr.normal(key, shape) for key,shape in zip(keys,shapes)]


def init_nn_params(key, arch: List[Tuple[int,int]]):
    nn = []

    keys = jr.split(key, len(arch))

    for i,layer in enumerate(arch):

        init_layer = random_layer(keys[i], layer[0], layer[1], affine=True if i != len(arch)-1 else False)

        nn.append(init_layer)

    return nn


def init_params(key, arch: List[Tuple[int,int]]):

    n_key, b_key = jr.split(key, 2)

    return init_nn_params(n_key, arch), init_batch_norm_parameters(b_key, arch)



# [B,K] -> [B,K]
def activations_batch_normalization(activations, eta=1e-6):
    batch_size = activations.shape[0]
    mean = activations.mean(axis=0) # (B,K) -> (K)
    var = jnp.var(activations, axis=0, ddof=0) # (B,K) -> (K,)
    x_i = (activations - mean[None,:])/jnp.pow(var[None,:] + eta, 0.5)

    return x_i, mean, var




def forward_pass(nn, batch_norm_params, image_vector):

    assert image_vector.ndim == 2

    output = image_vector

    means_ = []
    vars_ = []

    for i,(lt,bn) in enumerate(itertools.zip_longest(nn, batch_norm_params)):
        w = lt["weight"]

        output = output@w
        if lt.get("bias",None) is not None:
            output += lt["bias"]
        else:
            continue

        # activations
        x_i, mean, var = activations_batch_normalization(output)

        means_.append(mean)
        vars_.append(var)

        if bn is not None:
            output = x_i*bn[:,0][None,:] + bn[:,1][None,:]

        # Only now apply nonlinearity
        if i < len(nn) - 1:
            output = jax.nn.sigmoid(output)

    assert output.ndim == 2 #

    return output, means_, vars_ 


# can use vmap here
def forward_pass_inference(nn, batch_norm_params, running_stats, image_vector):

    # print(f"Batch Image vector Shape: {image_vector.shape}")

    assert image_vector.ndim == 1

    output = image_vector

    for i, (lt, bn, running_mean, running_var) in enumerate(itertools.zip_longest(nn, batch_norm_params, running_stats[0], running_stats[1])):

        w = lt["weight"]

        output = output@w
        if i < len(nn) - 1 and lt.get("bias",None) is not None:
            output += lt["bias"]
        else:
            continue

        if bn is not None:
            #print(f"output shape {output.shape} running_mean: {running_mean.shape} running_var: {running_var.shape}")
            x_i = (output - running_mean)/jnp.pow(running_var + 1e-6, 0.5)
            output = x_i*bn[:,0] + bn[:,1]

        if i < len(nn) - 1:
            output = jax.nn.sigmoid(output) 

         

    return output



def cross_entropy_loss(params, x,  labels):

    params, batch_norm_params  = params 
    #logits = jax.vmap(forward_pass, in_axes=(None,0))(params, x)
    logits, means, vars_ = forward_pass(params, batch_norm_params, x)

    # one-hot, this is batched
    logits = jax.nn.log_softmax(logits, axis=-1)
    loss = jax.vmap(operator.getitem, in_axes=(0,0))(logits, labels)
    return -loss.mean(), (means, vars_)


@jax.jit
def update(params, batch_norm_params, x, y):
    step_size = 0.01
    (vals, aux), grads = jax.value_and_grad(cross_entropy_loss, has_aux=True)((params, batch_norm_params), x , y)

    updated_params = [{"weight": params["weight"] - step_size * dparams["weight"], 
                       "bias": None if params.get("bias", None) == None else (params["bias"] - step_size * dparams["bias"]) } for params, dparams in zip(params, grads[0])]
    updated_batch_params =  [(param - step_size*dparam) for (param, dparam) in zip(batch_norm_params, grads[1])]

    return vals, grads, updated_params, updated_batch_params, aux[0], aux[1]


def accuracy(params, images, targets):
  target_class = jnp.argmax(targets, axis=1)
  predicted_class = jnp.argmax(jax.vmap(forward_pass, in_axes=(None,0))(params, images), axis=1)
  return jnp.mean(predicted_class == target_class)


arch = [(784,100), (100,100), (100,100), (100,10)]


def inf_dataloader(dataloader):
    while True:
        yield from dataloader


def evaluate_model(params, batch_params, running_stats):
    test_dataset = torchvision.datasets.MNIST("MNIST", train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=False)

    total_correct = 0
    total = 0

    for batch_x, batch_y in testloader:

        batch_x = batch_x.numpy().reshape(batch_x.shape[0], -1)
        preds = jax.vmap(forward_pass_inference, in_axes=(None, None, None, 0))(params, batch_params, running_stats, batch_x)
        predicted_class = jnp.argmax(preds, axis=-1)

        total_correct += (predicted_class == jnp.array(batch_y)).sum()
        total += batch_x.shape[0]

    acc = float(total_correct) / float(total)
    print(f"Evaluation accuracy: {acc:.4f}")
    return acc


# output type is List of jax.arrays
def calculate_running_stats(running_mean, running_var):

     running_mean = [jnp.stack(layer).mean(axis=0) for layer in zip(*running_mean)]
     running_var = [jnp.stack(layer).mean(axis=0) for layer in zip(*running_var)]

     return (running_mean, running_var)


steps = 2000
key = jr.PRNGKey(1232)
params, batch_norm_parameters = init_params(key, arch)

means_ , vars_ = [], []

for step, (batch_x, batch_y) in zip(range(steps), inf_dataloader(trainloader)):

    batch_x = batch_x.numpy().reshape(batch_x.shape[0],-1)
    batch_y = batch_y.numpy()

    loss, grads, params, batch_norm_parameters, means__, vars__ = update(params, batch_norm_parameters, batch_x, batch_y)

    means_.append(means__)
    vars_.append(vars__)

    if step % 100 == 0 :
        (running_mean, running_var) = calculate_running_stats(means_, vars_)
        print(f"Loss :{loss} for Step: {step}")
        evaluate_model(params, batch_norm_parameters, (running_mean, running_var))

    #print(batch_y)
