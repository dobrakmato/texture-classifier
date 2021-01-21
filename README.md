texture-classifier
---------------------

Currently, the imported textures may be tagged manually. Tags are then used to categorize imported textures and provide
search functionality in the asset browser part of the renderer.

We would like to automatically classify (predict tags)
textures that are imported into the renderer / asset browser project in Rust.

We have a dataset with ~1028 different textures which are labelled by ~461 different labels (~5952 total labels), so
this will be multi label classification task (the numbers are just approximate)

We would like to train and compare different neural network architectures and assess their performance (training time,
accuracy, runtime performance).

The trained model will be integrated into an application written in Rust. Unfortunately the dataset does not have
uniform distribution of labels as some labels (wooden) are more common than others (window), so we need to make sure to
split the dataset to training and test sets with similar distribution of labels.

### Usage

First you need to download the raw image data using `main.py`, then you need to pre-process the data
using `normalize.py` script and then you can train a neural network using the `cnn.py` script. Always check for possible
options (texture size, number of tags, pre-processing steps, cnn architecture) you can (and should) change in the
scripts, so they work together correctly.

### Scripts

- `main.py` - downloads the textures from [CC0 Textures](https://cc0textures.com/) along with tags into `textures`
  directory
- `normalize.py` - normalizes (pre-processes) the tags and textures to common resolution, color space, bit depth and
  format, produces 3 files (`imgs_XXX_Y.npy`, `labels_XXX_Y.npy` and `tag_mapping.json`)
- `normalize_combined.py` - normalizes (pre-processes) the tags and textures to common resolution, color space, bit
  depth and format but outputs training data with albedo and normals maps combined into 6 channel (RGB, RGB) training
  data
- `cnn.py` - trains a CNN using keras / tensorflow
- `tl_mobilenet.py` - trains a CNN from MobileNet CNN architecture with ImageNet weights

### Results

Report: <to be added later>

Weights and Biases Project: https://wandb.ai/dobrakmato/texture-classification
