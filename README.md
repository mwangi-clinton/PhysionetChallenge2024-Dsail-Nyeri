# Example entry to the 2024 PhysioNet Challenge

Authors: Zuzana Koscova, James Weigle

## Installing the dependencies

The following steps have been tested for Python 3.9 and Python 3.10.

It's probably a good idea to set up a fresh virtual environment:
```bash
python -m venv path/to/virtualenv
source path/to/virtualenv/bin/activate
```

## CPU-only

**Install PyTorch 2.1.1.**
For OS X, using Pip:
```bash
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1
```
For Linux and Windows, using Pip:
```bash
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cpu
```

See [this page](https://pytorch.org/get-started/previous-versions/) for more commands.

**Install requirements.txt.**

```bash
pip install -r requirements.txt
```

## GPU

**Make sure that CUDA is installed and note the CUDA version.**
On Linux, you can run the command `nvidia-smi`: in the upper left 
corner there should be a string like `CUDA Version: 12.3`.

**Run a matching version of the PyTorch 2.1.1 installation commands.**
[This page](https://pytorch.org/get-started/previous-versions/)
gives commands to install various versions of PyTorch
for various versions of CUDA. 
Look under `Wheel` and your operating system.
For example, if your Cuda version is 12.3
and your OS is Linux, you can use this command:

```bash
# CUDA 12.1
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu121
```

It's okay if the CUDA versions don't match exactly:
the PyTorch build for CUDA 12.1 still works for CUDA 12.3.
Try to get the same overall version number (e.g. 12).

**Install requirements.txt.**

```bash
pip install -r requirements.txt
```

## Want to use Docker instead?

Build with the Dockerfile that applies to your system, i.e. with or without a GPU.
For a system without a GPU:

```bash
docker build --file Dockerfile_cpu .
```
For a system with a GPU and CUDA 12:

```bash
docker build --file Dockerfile_gpu .
```

For a system with a GPU and another version of CUDA: edit the `FROM` line in `Dockerfile_gpu`
to use a different Pytorch base image.
You can find the appropriate one for your system [here](https://hub.docker.com/r/pytorch/pytorch/tags).

## Sources

This entry is based on the [Python example code](https://github.com/physionetchallenges/python-example-2024)
for the [2024 PhysioNet Challenge](https://moody-challenge.physionet.org/2024/).

### Data loader

The data classes for loading and processing the data are loosely based on the code
in [this repository](https://github.com/CN-zdy/AI-ECG-Image/tree/main).
