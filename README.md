<h1 align="center">🐌 NuMPItron</h1>

Simplistic small language model 3D-parallelism training using NumPy and MPI. Inspired by [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) and [Nanotron](https://github.com/huggingface/nanotron) and based only on [NumPy](https://numpy.org) and [MPI for Python](https://mpi4py.readthedocs.io), NuMPItron offers a variety of ways to train your Transformer at a snail's pace.

This library is meant as a learning experience for implementing distributed training strategies. Ideally the library will be capable of both 3D parallelism (TP + MP + DP) and ZeRO. If you want to follow along, make sure to check out my [blog](https://lweitkamp.github.io/).

# Feature Roadmap
Core functionality will be 3D parallel and ZeRO stage 1 since these can be combined in general:

* [x] Single Core 
* [ ] Tensor Parallel 
* [ ] Distributed Data Parallel 
* [ ] Pipeline Parallel
* [ ] ZeRO

When/if this is done, we can look at sequence parallel strategies.


# Installation
```bash
git clone https://github.com/lweitkamp/numpitron
cd numpitron
pip install -e .
```

If you want to additionally run the unit tests:
```bash
pip install -e .[dev]
pytest tests
```

# Examples
First, download the shakespare dataset (`shakespeare_char_{train|val}.bin`) from [Google Drive](https://drive.google.com/drive/folders/1VwFHJ8z7EmjTJZv4XsISTyPwwpELyMOs?usp=sharing) and place it in the `examples` folder.

You can run a sample character level training run on the shakespeare corpus using:
```bash
python train.py \
    --config-path examples/shakespeare_transformer.json \
    --save-path examples
```

This will save the parameters and optimizer state at `examples/shakespeare_Transformer.npy` to be used for sampling.

Be advised that training for about 10 epochs took 24+ hours on my 2015 macbook pro, with a loss of about ~1.80[^1].
I would not recommend training from scratch but to download the state `shakespeare_Transformer.npy` from [Google Drive](https://drive.google.com/drive/folders/1VwFHJ8z7EmjTJZv4XsISTyPwwpELyMOs?usp=sharing) to the `examples` folder.

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