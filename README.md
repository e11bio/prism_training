# prism_training
Example training scripts and models for PRISM paper

## Overview

[E11 Bio](https://www.e11.bio/) recently released a new technology - PRISM (Protein Reconstruction and Identification through Multiplexing), a platform that combines viral barcoding, expansion microscopy, and iterative immunolabeling for large-scale neuronal reconstruction. Neurons were labeled with combinatorial “protein bits” that act as barcodes to distinguish individual cells and support error-correction during reconstruction. 

Read more about the approach in the [paper](https://www.biorxiv.org/content/10.1101/2025.09.26.678648v1) and accompanying [blog post](https://www.e11.bio/blog/prism)

This is a simple tutorial for downloading and running the models used for neuron segmentation and synapse detection. This tutorial is currently pretty minimal, but will be extended/improved in the coming weeks. Additionally, all experimental code (including post-processing and analysis) will be released in a separate repository.

We uploaded data to a publically accessible [s3](https://aws.amazon.com/s3/) bucket via [aws open data](https://aws.amazon.com/opendata/). More details on the bucket contents can be seen in this [repository](https://github.com/e11bio/e11-open-data)

The reconstruction pipeline consists of 5 models:

1. barcode signal enhancement
2. affinities + LSDs
3. uniform embedding
4. barcode expression
5. synapse detection

This tutorial uses several different libraries for training/predicting/visualizing data including:

1. [gunpowder](https://funkelab.github.io/gunpowder/)
2. [dacapo](https://funkelab.github.io/dacapo/)
3. [volara](https://www.e11.bio/blog/volara)
4. [neuroglancer](https://github.com/google/neuroglancer)

## Getting started

Tested on ubuntu 22.04 with an a6000 gpu. Assumes basic python an ML knowledge. For some useful tutorial with affinity/lsd models see this [repo](https://github.com/funkelab/lsd)

1. clone this repo and create a conda environment + install packages:

```
git clone https://github.com/e11bio/prism_training.git; cd prism_training
conda create -n prism_training python=3.11
conda activate prism_training
pip install -e .
```

2. download and consolidate example data

```
cd prism_training/data  # from base directory
python download_data.py
python consolidate_data.py
cd ../../  # revert to base dir (optional)
```

3. predict enhanced data (takes about 10 minutes on NVIDIA RTX 6000 gpu)

```
cd prism_training/train/enhanced  # from base directory
python predict.py
cd ../../../  # revert to base dir (optional)
cp -r prism_training/data/instance/example_data.zarr/enhanced prism_training/data/semantic/example_data.zarr/enhanced
```

## Enhancement

![](https://github.com/e11bio/prism_training/blob/main/static/enhanced_batch.png)
![](https://github.com/e11bio/prism_training/blob/main/static/enhanced_pred.png)

## Affs/LSDs

![](https://github.com/e11bio/prism_training/blob/main/static/affs_batch.png)

## Uniform embedding

![](https://github.com/e11bio/prism_training/blob/main/static/uniform_batch.jpg)

## Barcode expression

![](https://github.com/e11bio/prism_training/blob/main/static/binary_batch.jpg)

## Synapses

![](https://github.com/e11bio/prism_training/blob/main/static/synapse_batch.png)
