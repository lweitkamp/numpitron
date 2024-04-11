import numpy as np

from numpitron import nn
from numpitron import distributed as npdist


class TensorParallelInputEmbedding(nn.InputEmbedding):
    """A Tensor Parallel embedding layer. The embedding table is split
    along the vocab dim - each process holds a chunk of the vocab.
    
    To ensure the next layer will get the full results of a single embedding
    table, we use a masking scheme that masks tokens not in the current
    process' chunk. An allreduce is required to fill in the masked locations."""

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        name="TensorParallelInputEmbedding",
        dtype=np.float32,
    ):
        super().__init__(
            d_model=d_model,
            vocab_size=vocab_size // npdist.tensor_parallel_size(),
            name=name,
            dtype=dtype,
        )

    def forward(
        self, params: dict[str, np.ndarray], inputs: np.ndarray
    ) -> tuple[dict, np.ndarray]:
        """Given a chunked embedding table and input tokens, embed the tokens.
        If the token indices are not found in the current process' chunk,
        mask them out with 0s in the output. The allreduce will ensure
        that each process fills in the 0s.

        Arguments:
            params (dict[str, np.ndarray]): parameters with key 'embedding'
                present.
            inputs (np.ndarray): Input tokens of type int32.

        Returns:
            Token embeddings.
        """
        # Figure out token valid range for this specific embedding chunk.
        chunk_start = npdist.tensor_parallel_rank() * params["embedding"].shape[1]
        chunk_end = chunk_start + params["embedding"].shape[1]
        mask = np.logical_or(inputs < chunk_start, inputs >= chunk_end)

        # Set tokens to chunk range, mask tokens outside range.
        inputs = inputs - chunk_start
        inputs[mask] = 0

        # Take the correct embeddings and mask outside range.
        inputs_embedding = np.take(params["embedding"].T, inputs, axis=0)
        inputs_embedding[mask, :] = 0.0

        npdist.all_reduce(inputs_embedding, group=npdist.tensor_parallel_group())
        ctx = {"inputs": inputs, "embedding": params["embedding"], "mask": mask}

        return ctx, inputs_embedding

    def backward(self, ctx: dict, d_out: np.ndarray) -> tuple[np.ndarray, dict]:
        g = np.zeros_like(ctx["embedding"])
        np.add.at(g.T, ctx["inputs"][~ctx["mask"]], d_out[~ctx["mask"]])
        return {"embedding": g}, d_out
