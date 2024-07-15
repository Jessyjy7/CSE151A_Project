# CSE151A_Project: Chinese MNIST Machine Learning Project
## You can find the detailed preprocessing steps in our [Google Colab Notebook](https://colab.research.google.com/drive/1f2XF-l_ncpJ0O94dz4d69kwemG1e9dMF#scrollTo=I29OQcEGtv6D).

## Data Download and Setup

Download the dataset from [Kaggle](https://www.kaggle.com/datasets/gpreda/chinese-mnist).

## Environment Setup

To set up the environment, run the following commands in your Jupyter Notebook:

```python
!pip install torch torchvision pandas matplotlib
!wget -O chinese_mnist.csv https://www.kaggle.com/datasets/gpreda/chinese-mnist/download

# Unzip the dataset if necessary
# !unzip path/to/your/downloaded/zipfile.zip -d target_directory
```

## Preprocessing Steps for Chinese MNIST Dataset
1. Data Download and Extraction
We download the dataset from a specified URL and extract the images into a directory. This step ensures all required data is available for preprocessing.

2. Data Loading
We load the dataset into a Pandas DataFrame to facilitate easy manipulation and access to the image metadata (such as suite_id, sample_id, and code).

3. Image Path Construction
For each image, we construct the file path using the metadata. This step ensures we can locate and process each image correctly.

4. Image Normalization
Normalization is a crucial preprocessing step. For this dataset, we normalize the pixel values to the range [0, 1] by dividing each pixel value by 255. This helps stabilize the learning process and improves model convergence.

5. Image Resizing (if necessary)
If the images are not of uniform size, we resize them to a consistent dimension (e.g., 64x64 pixels). Uniform image sizes are required for most machine learning models, especially convolutional neural networks (CNNs).

6. Handling Missing or Corrupted Data
During preprocessing, we check for missing or corrupted images. If any images are missing or cannot be opened, we log these instances and handle them appropriately (e.g., by skipping or replacing them).

