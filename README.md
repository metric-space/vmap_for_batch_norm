## What is this?

TLDR; Jax's vmap that forces you to think about map over the batch dimension would at first sight seem to run up against a wall when it comes to **Batch norm**.
The wall is an illusion. I was fooled. And here's my attempt to show that the wall isn't there


## Vmap, batch norm and more vamp

Jax is pretty sweet. It reeks of Haskell in a good way. It explicitly deals with what I call the "batch axis smell" i.e the batch axis stands out from the other axis and yet other frameworks expect you to treat it the same as the other axes.

What do I mean here? So assume we're solving the MNIST classification problem i.e given a black and white 28x28 image of a single digit, map it to a number from 0 to 9
It's a well defined problem.
Assume the neural net is a feed forward neural network which 3 hidden layers, 1 output layer. The hidden layer width is 100 i.e 100 activations per "weight-gap"

You have to write something that takes in a batch of images and pass it though this network. How do you proceed?

Important question: **At what stage do you think of the Batch Dimension?**

the JAX way: 

> the important step is figuring out how you'd want to deal with one image.
> Let the framework deal with the batch dimension. The framework's tool -> Vmap 

Here, look at the forward step and __look at the asserts__

```python

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

    assert output.ndim == 1 

    return output

```

and here's how the batch dimension is dealth with

```python

def cross_entropy_loss(params: NNParams , x: Float32[Array, "batch x"] , labels: Int32[Array, "batch z"]) -> Float32:

    logits = jax.vmap(forward_pass, in_axes=(None, 0))(params, x) # <- important bit
    # one-hot, this is batched
    logits = jax.nn.log_softmax(logits, axis=-1)
    loss = jax.vmap(operator.getitem, in_axes=(0, 0))(logits, labels)
    return -loss.mean()

```

During any trench run, Jax says "use the *vmap*, Luke!"


## How is Equinox connected to this?



## How to run the code

```bash

pip install -e .

time python scripts/main.py

```

## Implemementation output

[accuracy graph](./output.png)
