# CROWD: Multi-class Crowd Localization for Object Detection in Aerial Images

## Overview

We propose a transformer-based framework for multi-class object localization in dense remote sensing images. Based on [CLTR](https://github.com/dk-liang/CLTR), the framework uses a transformer architecture to predict object locations and class labels in aerial images containing objects at different scales and densities.

The project investigates multi-class localization using the VisDrone-DET and RSOC datasets. It also incorporates a dilated convolutional backbone to improve feature extraction across different object scales and uses classification and localization losses to train the model.

## Repository Layout

```
CLTR/        
data/      
mmcv_m/        
tests/     
```

## Requirements

- A CUDA-enabled Linux machine or cloud GPU instance (Ubuntu 22.04 LTS recommended), running as root
- At least 50 GB of free disk space

For Ubuntu, the common system packages can be installed with:

```bash
sudo apt update
sudo apt install -y python3.11-venv build-essential unzip
```

## Installation

Clone the repository:

```bash
git clone https://github.com/msprITU/CROWD.git
cd CROWD
```

Run these commands from the repository root:

```bash
chmod +x setup.sh
./setup.sh
source .venv/bin/activate
```

The script creates `.venv`, installs PyTorch and torchvision CUDA 12.4 wheels and Python dependencies, builds MMCV 2.1.0 with CUDA operations, and copies the our custom `mmcv_m/utils/` files into the installed MMCV package. 

If CUDA Toolkit 12.4 is installed somewhere other than `/usr/local/cuda-12.4`, provide its path explicitly.


## Dataset preparation

The preparation script downloads the VisDrone dataset as zip and processes it into the .h5 and .npy formats required for training. Run the following commands to prepare the data for training:

```bash
chmod +x prepare_data.sh
./prepare_data.sh
```
If you already have the dataset, ensure that the files are located at the expected paths or update the corresponding dataset paths in the scripts.

## Training

### Class Selection

The classes included in training are configured in `class_selection.py` Modify the class-selection configuration before training to determine which object categories are included in the experiment.

### Running Training

The following command demonstrates a training configuration:

```bash
python train.py \
    --num_queries 400 \
    --save \
    --batch_size 32 \
    --local_rank 0 \
    --threshold 0.35 \
    --test_per_epoch 300 \
    --dataset visdrone_dataset \
    --lr 25e-6 \
    --epochs 1500 \
    --lr_step 1200 \
    --dilation
```

Adjust the training arguments as needed for your experiment.

Refer to the argument definitions in `train.py` and the project's configuration files for the precise behavior of each option.

### Checkpoints

When checkpoint saving is enabled, the training script saves model files under a directory similar to:

```text
save_file/
└── log_file_single/
    └── <timestamp>/
        └── ...
```

The exact checkpoint filename and directory contents depend on the training implementation.

**Before running inference, locate the trained checkpoint (`.pth`) that you want to evaluate.** Copy its path into `tests/planner.txt`, following the format described in the next section.



## Inference

Inference runs trained models on a dataset and writes their predictions to output text files. These outputs can then be passed to the evaluation script.

### Selecting the Models

Open:

```text
tests/planner.txt
```

Add the path of each trained model checkpoint (`.pth`) on a separate line.

For example:

```text
save_file/log_file_single/experiment_1/checkpoint.pth
save_file/log_file_single/experiment_2/checkpoint.pth
save_file/log_file_single/experiment_3/checkpoint.pth
```

You can run inference for a single model or multiple models in one execution. 

### Running Inference

Run the inference script from the repository root.

For three models, each using 400 queries:

```bash
python tests/inference.py --queries_list 400 400 400
```

For a single model using 400 queries:

```bash
python tests/inference.py --queries_list 400
```

If the model is trained with dilated backbone, `--dilation` parameter should be set when running inference:

```bash
python tests/inference.py --queries_list 400 --dilation
```

By default, inference uses the test dataset. To run inference on the training dataset instead, change the dataset selection in `tests/inference.py` from the test dataset to the training dataset.

### Inference Outputs

The inference script saves its output files under:

```text
tests/inference_outputs/
```

Each model produces a corresponding inference output file, with a filename similar to:

```text
inference_<model_name>.txt
```

These output files contain the predictions required by the evaluation stage.

If your inference script does not automatically add generated output paths to `tests/eval_planner.txt`, add them manually as described below.

## Evaluation

The evaluation script reads inference output files, compares their predictions against the ground-truth annotations, and writes evaluation reports.

### Selecting Inference Outputs

Open:

```text
tests/eval_planner.txt
```

Add the paths of the inference output files you want to evaluate, with one path per line.

For example:

```text
inference_outputs/inference_model1.txt
inference_outputs/inference_model2.txt
inference_outputs/inference_model3.txt
```

Inference may automatically populate `eval_planner.txt` with the generated output paths. If it does not, edit the file manually.

### Configuring the Evaluation Classes

**The class-selection configuration used during evaluation must match the configuration used to train the model.**

In `tests/evaluation.py`, find the class-selection block containing assignments of the form:

```python
k[k == X] = Y
```

These assignments determine which class labels are retained during evaluation:

- Set the selected classes' labels to their original class IDs.
- Set the excluded classes' labels to `0`.

For example, the following configuration retains classes `1` and `5` while excluding the other listed classes:

```python
k[k == 1] = 1
k[k == 2] = 0
k[k == 3] = 0
k[k == 4] = 0
k[k == 5] = 5
k[k == 6] = 0
k[k == 7] = 0
k[k == 8] = 0
```

**Do not evaluate models trained with different class selections together under one shared evaluation configuration.**

### Running Evaluation

From the repository root, run:

```bash
python tests/evaluation.py
```

The script processes the inference output files listed in `tests/eval_planner.txt` sequentially.

### Evaluation Results

Evaluation reports are saved under:

```text
tests/eval_results/
```

The generated reports use filenames similar to:

```text
result_<model_name>.txt
```

The reports contain the metrics calculated by the evaluation script.


## Acknowledgements

This project builds on the CLTR framework:

- **CLTR:** [Counting with Transformer](https://github.com/dk-liang/CLTR)

Please refer to the original repository for details of the underlying framework and its implementation.

## Citation

...
