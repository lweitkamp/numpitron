<h1 align="center">🐌 NuMPItron</h1>

Simplistic small language model 3D-parallelism training using NumPy and MPI. Inspired by [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) and [Nanotron](https://github.com/huggingface/nanotron) and based only on [NumPy](https://numpy.org) and [MPI for Python](https://mpi4py.readthedocs.io), NuMPItron offers a variety of ways to train your Transformer at a snail's pace.

This library is meant as a learning experience for implementing distributed training strategies. Each added strategy will be discussed in detail in blog posts.

# Core Features

We support the following:
- Single Core Training

Next on the list is:
- Tensor Parallelization (Megatron-LM style)

We will soon support the following:
- DataParallel
- ZeRO


# Installation
```bash
git clone git@github.com:lweitkamp/numpitron.git
cd numpitron
pip install -e .
```

# Examples
First, download the shakespare dataset: and place it in the `examples` folder.

You can run a sample character level training run on the shakespeare corpus using:
```bash
python train.py --config-path examples/shakespeare_transformer.json --save-path examples
```

This will save the parameters and optimizer state at `examples/shakespeare_Transformer.npy` to be used for sampling.

Be advised that training for about 10 epochs took 24+ hours on my 2015 macbook pro, with a loss of about ~1.80[^1].
I would not recommend training from scratch but to download the state from [google drive URL] to the `examples` folder.

Run a sample generation using:
```bash
python sample.py --config-path examples/shakespeare_transformer.json --state-path examples/shakespeare_Transformer.npy
```

With the pretrained model loaded you would expect to see the following text below. Not bad, not great.

```
Somaging:
I am as I, Wath I drows Bolingbourable is the equittion our to me housand;
My sound, there the speech your thether is
What is blessixes, gard carrer are prince of All,
Has enluckin. Theer betther,
And live might! this subjectt
to fill they
```

[^1]: This matches Karpathy's log loss at same model size at his [NanoGPT](https://github.com/karpathy/nanoGPT?tab=readme-ov-file#quick-start) repo.