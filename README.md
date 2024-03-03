# 🐌 NuMPItron

Simplistic small language model 3D-parallelism training using NumPy and MPI. Inspired by [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) and [Nanotron](https://github.com/huggingface/nanotron) and based only on [NumPy](https://numpy.org) and [MPI for Python](https://mpi4py.readthedocs.io), NuMPItron offers a variety of ways to train your Transformer at a snail's pace.

# Core Features

We support the following:
- Single Core Training
- Tensor Parallelism

# Installation
```bash
git clone git@github.com:lweitkamp/numpitron.git
cd numpitron
pip install -e .
```

# Examples

You can run a sample character level training run on the shakespeare corpus using:
```bash
python train.py --save_dir examples/model.weights
```

And run a sample generation using:
```bash
python sample.py --model_dir examples/model.weights
```
