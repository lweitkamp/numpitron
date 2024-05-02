import numpy as np
import pytest
import torch
from torch import nn

from numpitron import distributed as npdist
from numpitron.nn import softmax_cross_entropy

npdist.init(tp_size=npdist.world_size())


@pytest.mark.skipif(npdist.world_size() != 1, reason="Cannot be run in TP mode.")
def test_softmax_cross_entropy_pytorch():
    b, s, v = 32, 64, 16

    rng = np.random.default_rng(42)

    inputs = rng.random((b, s, v))
    inputs_torch = torch.from_numpy(inputs).reshape(b * s, -1)
    inputs_torch.requires_grad = True

    labels = rng.integers(0, v, (b, s))
    labels_torch = torch.from_numpy(labels).reshape(-1)

    out, d_out = softmax_cross_entropy(inputs, labels)
    out_torch = nn.CrossEntropyLoss(reduction="none")(inputs_torch, labels_torch)

    out_torch.sum().backward()

    np.testing.assert_allclose(out.reshape(-1), out_torch.detach().numpy())
    np.testing.assert_allclose(
        d_out.reshape(b * s, v), inputs_torch.grad.detach().numpy()
    )


@pytest.mark.skipif(npdist.world_size() != 2, reason="Requires MPI with two processes.")
def test_tensor_parallel_softmax_cross_entropy_pytorch():
    b, s, v = 32, 64, 16

    rng = np.random.default_rng(42)

    inputs = rng.random((b, s, v))
    inputs_torch = torch.from_numpy(inputs).reshape(b * s, -1)
    inputs_torch.requires_grad = True

    labels = rng.integers(0, v, (b, s))
    labels_torch = torch.from_numpy(labels).reshape(-1)

    inputs_scattered = np.zeros((b, s, v // npdist.tensor_parallel_size()))
    npdist.scatter(
        inputs, inputs_scattered, axis=-1, group=npdist.tensor_parallel_group()
    )
    print(inputs_scattered.dtype)
    out, d_out = softmax_cross_entropy(inputs_scattered, labels)
    out_torch = nn.CrossEntropyLoss(reduction="none")(inputs_torch, labels_torch)

    out_torch.sum().backward()

    d_out_gathered = np.zeros((b, s, v))
    npdist.all_gather(
        d_out, d_out_gathered, axis=-1, group=npdist.tensor_parallel_group()
    )

    np.testing.assert_allclose(out.reshape(-1), out_torch.detach().numpy())
    np.testing.assert_allclose(
        d_out_gathered.reshape(b * s, v), inputs_torch.grad.detach().numpy()
    )
