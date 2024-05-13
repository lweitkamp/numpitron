<h1 align="center">🐌 NuMPItron</h1>

Simplistic small language model 3D-parallelism training using NumPy and MPI. Inspired by [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) and [Nanotron](https://github.com/huggingface/nanotron) and based only on [NumPy](https://numpy.org) and [MPI for Python](https://mpi4py.readthedocs.io), NuMPItron offers a variety of ways to train your Transformer at a snail's pace.

This library is meant as a learning experience for implementing distributed training strategies. Ideally the library will be capable of both 3D parallelism (TP + MP + DP) and ZeRO. If you want to follow along, make sure to check out my [blog](https://lweitkamp.github.io/).

# Feature Roadmap
Core functionality will be 3D parallel and ZeRO stage 1 since these can be combined in general:

* [x] Single Core 
* [x] Tensor Parallel 
* [ ] Distributed Data Parallel 
* [ ] Pipeline Parallel
* [ ] ZeRO

When/if this is done, we will look at expert parallel strategies.


# Installation
First, ensure `mpi4py` is installed by following the instructions on the [MPI for Python](https://mpi4py.readthedocs.io) page.

Then, install the library using:

```bash
git clone https://github.com/lweitkamp/numpitron
cd numpitron
pip install -e .  # -e .[dev] for unit tests
```

# Examples
You will need to download the shakespeare dataset (`shakespeare_char_{train|val}.bin`) from [Google Drive](https://drive.google.com/drive/folders/1VwFHJ8z7EmjTJZv4XsISTyPwwpELyMOs?usp=sharing) and place it in the `examples` folder.

Training with tensor parallelism can be done using the `train_shakespeare.py` script:
```bash
mpirun -n 2 python train_shakespeare.py --tensor-parallel-size 2
```

This will also save the parameters and optimizer state at `examples/model.npy` to be used for sampling. Training can take up to two days for 100 epochs but you can get a pretty good model at ~1.80[^1] validation loss within 5 hours or so, depending on your hardware (I'm using a 2015 macbook pro).

Run a sample generation using:
```bash
python sample.py \
    --config-path examples/shakespeare_transformer.json \
    --state-path examples/shakespeare_Transformer.npy
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
