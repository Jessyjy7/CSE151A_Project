# CSE151A_Project: Chinese MNIST Machine Learning Project
## You can find the detailed preprocessing steps in our [Google Colab Notebook](https://colab.research.google.com/drive/1f2XF-l_ncpJ0O94dz4d69kwemG1e9dMF#scrollTo=I29OQcEGtv6D).

## Project Overview

Our primary objective is to develop and evaluate models that can accurately recognize and classify handwritten Chinese numerals. We employ various machine learning algorithms to handle the image data, leveraging their capability to capture spatial hierarchies in visual patterns.

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

5. Image Resizing
We resize the images to a consistent dimension (e.g., 32x32 pixels). Uniform image sizes are required for most machine learning models. This step is crucial because it made sure that we won't use all the RAM. 

6. Handling Missing or Corrupted Data
During preprocessing, we check for missing or corrupted images. If any images are missing or cannot be opened, we log these instances and handle them appropriately (e.g., by skipping or replacing them).

7. Feature Expansion
We apply polynomial feature expansion with a lower degree to generate new features from the existing ones, improving the model's ability to capture complex patterns in the data.

## Model Training

1. Incremental Learning
We train an SGD classifier using incremental learning to manage memory usage effectively.

2. Train-Test Split
We split the dataset into training and testing sets to evaluate the model's performance on unseen data.

3. Model Evaluation
We evaluate the model using classification reports and confusion matrices.

Fitting Graph
We plot the training and test accuracies over batches to visualize the model's performance and understand where it fits in terms of underfitting, overfitting, or well-fitted. 

4. Conclusion and Next Steps

Conclusion of the First Model
- **Training Accuracy**: The model achieves good training accuracy (avg: 70%), indicating that it learns the training data well.
- **Test Accuracy**: The test accuracy is significantly lower (around 50-55%), suggesting that the model struggles to generalize to unseen data.
- **Overfitting**: The substantial gap between training and test accuracy indicates that the model is overfitting. It performs well on the training data but fails to generalize to new, unseen data.

Possible Improvements
To address the overfitting issue and improve the model's performance, consider the following strategies:

1. **Regularization Techniques**:
   - Adding L2 regularization to penalize large weights.
   - Using dropout layers to prevent overfitting in neural networks.

2. **More Complex Models**:
   - Exploring Convolutional Neural Networks (CNNs) which are well-suited for image classification tasks.

4. **Cross-Validations**:
   - Use cross-validation to ensure that the model's performance is consistent across different subsets of the data.
