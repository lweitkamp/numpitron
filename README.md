# 🐌 NuMPItron

Simplistic small language model 3D-parallelism training using NumPy and MPI. Inspired by [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) and [Nanotron](https://github.com/huggingface/nanotron) and based only on [NumPy](https://numpy.org) and [MPI for Python](https://mpi4py.readthedocs.io), NuMPItron offers a variety of ways to train your Transformer at a snail's pace.

# Core Features

We support the following:
- Single Core Training
- Tensor Parallelization (Megatron-LM style)

We will soon support the following:
- Sequence Parallelization (Megatron-LM style)
- DataParallel

And on the roadmap is:
- Ring Attention
- ZeRO



# Installation
```bash
git clone git@github.com:lweitkamp/numpitron.git
cd numpitron
pip install -e .
```

# Examples

You can run a sample character level training run on the shakespeare corpus using:
```bash
python train.py --save_dir examples
```

This will save both the model (`examples/model.state`) and the optimizer (`examples/optimizer.state`) to be used for sampling.
Be advised that training for about 10 epochs took 24+ hours on my 2015 macbook pro, with a loss of about ~1.80.
I would not recommend training from scratch but to download  the states from [google drive URL] to the `examples` folder.

And run a sample generation using:
```bash
python sample.py --model_dir examples/model.state
```

With the state loaded from google drive we can generate the following text:

```
DUSHAM:
In the good igratfus' hast. Arbly shink? it; leeave
To himan in peearin: hand provone exfor!

KING bractals an the sa a have my seee hear:
Theit belse.

DUKE VINCENTIUS:
Your time?

DUKE VINCENTIO:
Mhers acre:
Jore love shoried you shakle oS:
Of Enord, my the the will teant:
Onphy our shall Forth;
Andseruld, must laught, in a devommethaing pare be ans
Int wa so fortued wint mys thee on holl you The rifen quaught,
This his ruch dand you palk mee ro.

Maresed or ble spake the and it ters.
```