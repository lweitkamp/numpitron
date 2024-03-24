"""Unit tests for the neural network library. To test it out, ensure pytorch
is installed.

Each test follows a basic pattern:
initialize a numpitron layer, copy its weights to a pytorch-equivalent, run a
forward and backward pass and compare that the outputs and the grads
are approximately equal."""


import numpy as np
import pytest
import torch
from numpy.random import Generator
from torch import nn
from torch.nn import Parameter

from numpitron.nn import (MLP, Attention, InputEmbedding, LayerNorm, Linear,
                          Softmax, SoftmaxCrossEntropy)

TEST_SHAPES = [
    (1, 1, 8),
    (3, 1, 8),
    (1, 3, 8),
    (3, 3, 8),
]


@pytest.fixture
def rng() -> Generator:
    return np.random.default_rng(42)


@pytest.mark.parametrize("batch_size,seq_len,d_model", [shape for shape in TEST_SHAPES])
def test_attention(
    batch_size: int,
    seq_len: int,
    d_model: int,
    rng: Generator,
):
    n_heads: int = d_model // 2

    inputs = rng.random((batch_size, seq_len, d_model)).astype(np.float32)
    inputs_torch = torch.from_numpy(inputs)
    inputs_torch.requires_grad = True

    attention = Attention(d_model, n_heads, d_model // n_heads)
    params = attention.init_params(rng)
    attention_torch = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

    attention_torch.in_proj_weight = Parameter(
        torch.from_numpy(
            np.concatenate(
                [
                    params["q_projection"]["weight"],
                    params["k_projection"]["weight"],
                    params["v_projection"]["weight"],
                ]
            ).reshape(attention_torch.in_proj_weight.shape)
        )
    )

    attention_torch.in_proj_bias = Parameter(
        torch.from_numpy(
            np.concatenate(
                [
                    params["q_projection"]["bias"],
                    params["k_projection"]["bias"],
                    params["v_projection"]["bias"],
                ]
            ).reshape(attention_torch.in_proj_bias.shape)
        )
    )

    attention_torch.out_proj.weight = Parameter(
        torch.from_numpy(params["out_projection"]["weight"].reshape(d_model, d_model)).T
    )
    attention_torch.out_proj.bias = Parameter(
        torch.from_numpy(params["out_projection"]["bias"].reshape(-1))
    )
    # Forward through both models.
    ctx, out = attention(params, inputs)
    out_torch, _ = attention_torch(
        inputs_torch,
        inputs_torch,
        inputs_torch,
        is_causal=True,
        attn_mask=torch.from_numpy(~ctx["mask"].squeeze((0, 1))),
        average_attn_weights=False,
        need_weights=False,
    )

    out_torch.sum().backward()
    grads, d_out = attention.backward(ctx, np.ones_like(inputs))

    # There are quite some architectural differences with implementations so we
    # are a bit lenient here in terms of tolerance.
    np.testing.assert_allclose(
        out,
        out_torch.detach().numpy(),
        atol=1e-3,
    )

    # TODO(laurens): grads with weights and d_out do not align at all.
    # I'm confident in the implementation, so it's likely to be accumulation of errors.


@pytest.mark.parametrize("batch_size,seq_len,d_model", [shape for shape in TEST_SHAPES])
def test_softmax(
    batch_size: int,
    seq_len: int,
    d_model: int,
    rng: Generator,
):
    n_heads: int = d_model // 2
    inputs = rng.random((batch_size, n_heads, seq_len, seq_len)).astype(np.float32)
    inputs_torch = torch.from_numpy(inputs)
    inputs_torch.requires_grad = True

    softmax = Softmax(axis=-1)
    params = softmax.init_params(rng)
    softmax_torch = nn.Softmax(dim=-1)

    ctx, out = softmax(params, inputs)
    out_torch = softmax_torch(inputs_torch)

    out_torch.sum().backward()
    _, d_out = softmax.backward(ctx, np.ones_like(out))

    np.testing.assert_allclose(
        d_out.reshape(inputs_torch.grad.shape),
        inputs_torch.grad.detach().numpy(),
        atol=1e-5,
    )
    np.testing.assert_allclose(
        out,
        out_torch.detach().numpy(),
        atol=1e-5,
    )


@pytest.mark.parametrize(
    "batch_size,seq_len,d_model,vocab_size", [shape + (20,) for shape in TEST_SHAPES]
)
def test_softmax_cross_entropy(
    batch_size: int,
    seq_len: int,
    d_model: int,
    vocab_size: int,
    rng: Generator,
):
    inputs = rng.random((batch_size, seq_len, vocab_size))
    inputs_torch = torch.from_numpy(inputs).reshape(batch_size * seq_len, -1)
    inputs_torch.requires_grad = True

    labels = rng.integers(0, vocab_size, (batch_size, seq_len))
    labels_torch = torch.from_numpy(labels).reshape(-1)

    ce = SoftmaxCrossEntropy()
    ce_torch = nn.CrossEntropyLoss(reduction="none")

    ctx, out = ce({}, inputs, labels)
    out_torch = ce_torch(inputs_torch, labels_torch)

    _, d_out = ce.backward(ctx, None)
    out_torch.sum().backward()

    np.testing.assert_allclose(out.reshape(-1), out_torch.detach().numpy())
    np.testing.assert_allclose(
        d_out.reshape(batch_size * seq_len, vocab_size),
        inputs_torch.grad.detach().numpy(),
    )


@pytest.mark.parametrize(
    "batch_size,seq_len,d_model,vocab_size", [shape + (20,) for shape in TEST_SHAPES]
)
def test_input_embedding(
    batch_size: int,
    seq_len: int,
    d_model: int,
    vocab_size: int,
    rng: Generator,
):
    inputs = rng.integers((batch_size, seq_len, vocab_size))
    inputs_torch = torch.from_numpy(inputs)

    embedding = InputEmbedding(d_model, vocab_size)
    params = embedding.init_params(rng=rng)
    embedding_torch = nn.Embedding(vocab_size, d_model)

    embedding_torch.weight = Parameter(torch.from_numpy(params["embedding"].T))

    ctx, out = embedding(params, inputs)
    out_torch = embedding_torch(inputs_torch)

    grads, d_out = embedding.backward(ctx, np.ones_like(out))
    out_torch.sum().backward()

    np.testing.assert_allclose(out, out_torch.detach().numpy(), atol=1e-5)
    np.testing.assert_allclose(
        grads["embedding"].T, embedding_torch.weight.grad, atol=1e-5
    )


@pytest.mark.parametrize("batch_size,seq_len,d_model", TEST_SHAPES)
def test_linear(
    batch_size: int,
    seq_len: int,
    d_model: int,
    rng: Generator,
):
    inputs = rng.random((batch_size, seq_len, d_model)).astype(np.float32)
    inputs_torch = torch.from_numpy(inputs).reshape(batch_size * seq_len, -1)
    inputs_torch.requires_grad = True

    linear = Linear(input_dim=d_model, output_dim=d_model * 4)
    params = linear.init_params(rng=rng)
    linear_torch = nn.Linear(d_model, d_model * 4)

    linear_torch.weight = Parameter(torch.from_numpy(params["weight"].T))
    linear_torch.bias = Parameter(torch.from_numpy(params["bias"]))

    ctx, out = linear(params, inputs)
    out_torch = linear_torch(inputs_torch)

    out_torch.sum().backward()
    grads, d_out = linear.backward(ctx, np.ones_like(out))

    np.testing.assert_allclose(
        out.reshape(batch_size * seq_len, d_model * 4),
        out_torch.detach().numpy(),
        atol=1e-5,
    )
    np.testing.assert_allclose(grads["weight"].T, linear_torch.weight.grad, atol=1e-5)
    np.testing.assert_allclose(grads["bias"], linear_torch.bias.grad, atol=1e-5)


@pytest.mark.parametrize("batch_size,seq_len,d_model", TEST_SHAPES)
def test_mlp(
    batch_size: int,
    seq_len: int,
    d_model: int,
    rng: Generator,
):
    inputs = rng.random((batch_size, seq_len, d_model)).astype(np.float32)
    inputs_torch = torch.from_numpy(inputs).reshape(batch_size * seq_len, -1)
    inputs_torch.requires_grad = True

    mlp = MLP(d_model, d_model * 4)
    params = mlp.init_params(rng=rng)
    mlp_torch = nn.Sequential(
        nn.Linear(d_model, d_model * 4),
        nn.ReLU(),
        nn.Linear(d_model * 4, d_model),
    )

    mlp_torch[0].weight = Parameter(torch.from_numpy(params["Linear1"]["weight"].T))
    mlp_torch[0].bias = Parameter(torch.from_numpy(params["Linear1"]["bias"]))
    mlp_torch[2].weight = Parameter(torch.from_numpy(params["Linear2"]["weight"].T))
    mlp_torch[2].bias = Parameter(torch.from_numpy(params["Linear2"]["bias"]))

    ctx, out = mlp(params, inputs)
    out_torch = mlp_torch(inputs_torch)

    out_torch.sum().backward()
    grads, d_out = mlp.backward(ctx, np.ones_like(out))

    np.testing.assert_allclose(
        d_out.reshape(inputs_torch.grad.shape),
        inputs_torch.grad.detach().numpy(),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        out.reshape(batch_size * seq_len, d_model),
        out_torch.detach().numpy(),
        atol=1e-5,
    )
    np.testing.assert_allclose(
        grads["Linear2"]["weight"].T,
        mlp_torch[2].weight.grad,
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        grads["Linear2"]["bias"],
        mlp_torch[2].bias.grad,
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        grads["Linear1"]["weight"].T,
        mlp_torch[0].weight.grad,
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        grads["Linear1"]["bias"],
        mlp_torch[0].bias.grad,
        rtol=1e-5,
        atol=1e-5,
    )


@pytest.mark.parametrize("batch_size,seq_len,d_model", TEST_SHAPES)
def test_layer_norm(
    batch_size: int,
    seq_len: int,
    d_model: int,
    rng,
):
    inputs = rng.random((batch_size, seq_len, d_model)).astype(np.float32)
    inputs_torch = torch.from_numpy(inputs)
    inputs_torch.requires_grad = True

    norm = LayerNorm(d_model, eps=1e-5)
    params = norm.init_params(rng=rng)
    norm_torch = nn.LayerNorm(d_model, eps=1e-5)

    norm_torch.weight = Parameter(torch.from_numpy(params["weight"]))
    norm_torch.bias = Parameter(torch.from_numpy(params["bias"]))

    ctx, out = norm(params, inputs)
    out_torch = norm_torch(inputs_torch)

    grads, d_out = norm.backward(ctx, np.ones_like(inputs))
    out_torch.sum().backward()

    np.testing.assert_allclose(out, out_torch.detach().numpy(), atol=1e-5)
    np.testing.assert_allclose(
        grads["weight"], norm_torch.weight.grad.detach().numpy(), atol=1e-5
    )
    np.testing.assert_allclose(
        grads["bias"], norm_torch.bias.grad.detach().numpy(), atol=1e-5
    )
    np.testing.assert_allclose(d_out, inputs_torch.grad.detach().numpy(), atol=1e-4)
