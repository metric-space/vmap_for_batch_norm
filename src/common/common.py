from typing import Tuple, List, Dict, Any

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from jaxtyping import Pytree, Float32
from collections.abc import Callable



# ======== Types ===============

type ParamType = Array[Float, "w h"]
type NNParams = List[Dict[str, ParamType]]
type BatchParams = List[ParamType]
type ParamsType = Union[NNParams, Tuple[NNParams, BatchParams]]

# =============================

def named_grad_norms(grads) -> str:
    """

    Tool to list and pretty print gradient flow through the model

    """
    flat = jax.tree_util.tree_flatten_with_path(grads)[0]
    return {
        ".".join(str(k) for k in path): jnp.sqrt(jnp.sum(leaf**2))
        for path, leaf in flat if leaf is not None
        }


def dataset(train: bool = True, batch_size: int=100) -> DataLoader:
    transform = transforms.Compose([ transforms.ToTensor() ])
    dataset = torchvision.datasets.MNIST("MNIST", train=train, download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return dataloader



def inf_dataloader(dataloader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
    while True:
        yield from dataload


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



# TODO: check if typing makes sense
def evaluate_model(forward_pass: Callable[[Params, Array[Float, "batch flattened_images"]],  Array[Float, "batch predictions"]], predictions, **kwargs):

    testloader = dataset(False, batch_size=10)

    total_correct = 0
    total = 0

    for batch_x, batch_y in testloader:

        batch_x = batch_x.numpy().reshape(batch_x.shape[0], -1)
        preds = forward_pass(**kwargs)
        predicted_class = jnp.argmax(preds, axis=-1)

        total_correct += (predicted_class == jnp.array(batch_y)).sum()
        total += batch_x.shape[0]

    acc = total_correct / total

    # TODO: eliminate this 
    print(f"Evaluation accuracy: {acc:.4f}")
    return acc
