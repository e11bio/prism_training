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

> ## Pre-requisites
> We highly recommend using a package manager. `conda`, `virtualenv`, or `uv` are all good examples. The instructions are
> created assuming usage of `uv`. [Here](https://docs.astral.sh/uv/getting-started/installation/) are the installation instructions.
>
> Tested on ubuntu 22.04 with an a6000 gpu. Assumes basic python an ML knowledge. For some useful tutorial with affinity/lsd models see this [repo](https://github.com/funkelab/lsd)
---


1. clone this repo:
    ```
    git clone https://github.com/e11bio/prism_training.git
    cd prism_training/prism_training
    ```

2. download and consolidate example data
    ```
    cd data  # from script directory
    uv run download_data.py
    uv run consolidate_data.py
    cd ../  # revert to script directory (optional)
    ```

3. predict enhanced data (takes about 10 minutes on NVIDIA RTX 6000 gpu)
    ```
    cd train/enhanced  # from script directory
    uv run predict.py
    cd ../../  # revert to script directory (optional)
    cp -r prism_training/data/instance/example_data.zarr/enhanced prism_training/data/semantic/example_data.zarr/enhanced
    ```

4. run any of the other models via `uv run train.py`. Some models take arguments, please read the individual README's.

## Enhancement

* Example training from scratch for 10 iters: `python train.py -i 10`

Since we by default compute the difference between the average barcodes and the raw data as our target signal, a batch might look like:

![](https://github.com/e11bio/prism_training/blob/main/static/example_diff_barcodes.png)

Might have to tweak the shader a bit to see the target since it can contain negative values. The black pixels around the object denote the sparsely masked label for training (pixels outside of this label do not contribute to the loss). No need to visualize the predictions yet since this is from scratch so they will be uninformative.

* Example training from scratch and learning the direct average barcodes rather than residuals: `python train.py -d false`

A batch might then look like:

![](https://github.com/e11bio/prism_training/blob/main/static/example_avg_barcodes.png)

Which is a bit more visually intuitive. 

* Example training from downloaded checkpoint: `python train.py -c model`

Now we can visualize the predictions (residual barcode), and we can visualize the predicted average barcodes (simply adding the residual to the raw data). A batch might then look like:

![](https://github.com/e11bio/prism_training/blob/main/static/example_diff_added_barcodes.png)

If we then run inference, i.e `python predict.py` and visualize the raw vs enhanced, we could see something like: 

![](https://github.com/e11bio/prism_training/blob/main/static/enhanced_pred.png)

This is using a more fancy custom shader in which each channel is percentile normalized first.

## Affs/LSDs

Example training from model using raw input: `python train.py -d raw -c model`

A batch might look like:

![](https://github.com/e11bio/prism_training/blob/main/static/example_raw_affs_lsds.png)

The predictions are kind of noisy since the raw data is used as input.

Assuming we ran enhancement inference above, example using enhanced input: `python train.py -d enhanced -c model`

Which might give us something cleaner like:

![](https://github.com/e11bio/prism_training/blob/main/static/example_enhanced_affs_lsds.png)

## Uniform embedding

![](https://github.com/e11bio/prism_training/blob/main/static/uniform_batch.jpg)

## Barcode expression

![](https://github.com/e11bio/prism_training/blob/main/static/binary_batch.jpg)

## Synapses

Example training from model: `python train.py -c model`

A batch might look like:

![](https://github.com/e11bio/prism_training/blob/main/static/synapse_batch.png)
