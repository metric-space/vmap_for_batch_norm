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


# Goal: Batch norm for inference: understanding the state carrying part
#
# NN: 3 hidden layers
# o

config = dict(batch_size=100)


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




def init_nn_params(key, arch: List[Tuple[int,int]]):
    nn = []

    keys = jr.split(key, len(arch))

    for i,layer in enumerate(arch):

        init_layer = random_layer(keys[i], layer[0], layer[1], affine=True if i != len(arch)-1 else False)

        nn.append(init_layer)

    return nn



def forward_pass(nn, image_vector):

    assert image_vector.ndim == 1

    output = image_vector

    for i,lt in enumerate(nn):
        w = lt["weight"]

        output = output@w
        if lt.get("bias",None) is not None:
            output += lt["bias"]
        else:
            continue
        output = jax.nn.sigmoid(output)

    assert output.ndim == 1 #

    return output



def cross_entropy_loss(params, x,  labels):

    logits = jax.vmap(forward_pass, in_axes=(None,0))(params, x)
    # one-hot, this is batched
    logits = jax.nn.log_softmax(logits, axis=-1)
    loss = jax.vmap(operator.getitem, in_axes=(0,0))(logits, labels)
    return -loss.mean()


@jax.jit
def update(params, x, y):
    step_size = 0.01
    vals, grads = jax.value_and_grad(cross_entropy_loss)(params, x , y)

    updated_params = [{
        "weight": params["weight"] - step_size * dparams["weight"], 
        "bias": None if params.get("bias", None) == None else (params["bias"] - step_size * dparams["bias"])
        } for params, dparams in zip(params, grads)]

    return vals, grads, updated_params


def accuracy(params, images, targets):
  target_class = jnp.argmax(targets, axis=1)
  predicted_class = jnp.argmax(jax.vmap(forward_pass, in_axes=(None,0))(params, images), axis=1)
  return jnp.mean(predicted_class == target_class)


arch = [(784,100), (100,100), (100,100), (100,10)]


def inf_dataloader(dataloader):
    while True:
        yield from dataloader


def evaluate_model(params):
    test_dataset = torchvision.datasets.MNIST("MNIST", train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=False)

    total_correct = 0
    total = 0

    for batch_x, batch_y in testloader:

        batch_x = batch_x.numpy().reshape(batch_x.shape[0], -1)
        preds = jax.vmap(forward_pass, in_axes=(None, 0))(params, batch_x)
        predicted_class = jnp.argmax(preds, axis=-1)

        total_correct += (predicted_class == jnp.array(batch_y)).sum()
        total += batch_x.shape[0]

    acc = total_correct / total
    print(f"Evaluation accuracy: {acc:.4f}")
    return acc



steps = 2000
key = jr.PRNGKey(1232)
params = init_nn_params(key, arch)

for step, (batch_x, batch_y) in zip(range(steps), inf_dataloader(trainloader)):

    batch_x = batch_x.numpy().reshape(batch_x.shape[0],-1)
    batch_y = batch_y.numpy()

    loss, grads, params = update(params, batch_x, batch_y)

    if step % 100 == 0 :
        print(f"Loss :{loss} for Step: {step}")
        evaluate_model(params)

    #print(batch_y)

