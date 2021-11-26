# Multi-output ConvCNP

An implementation of using the conditional convolutional neural process (ConvCNP) for multiple continuous and discrete valued outputs. Based on the initial implementation by Wessel Bruinsma of a [dual ConvCNP](https://github.com/wesselb/gabriel-convcnp).

## Installation

1. Clone and enter the repo.

```bash
git clone https://github.com/ZGurney/iib-project
cd iib-project/code/multi-convcnp
```

2. Make and activate a virtual environment.

```bash
virtualenv -p python3.8 venv
source venv/bin/activate
```

3. Install [PyTorch](https://pytorch.org/), using a GPU-accelerated version if have access to CUDA capability.

4. Finally, install the project from `setup.py`.

```bash
pip install -e .
```

## Training the ConvCNP

Run

```bash
python train.py
```

For more information, see

```bash
python train.py --help
```

By default, results will be produced in `outputs`, but you can change that by setting `--root some/other/path`.

