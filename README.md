# CSE151A_Project: Chinese MNIST Machine Learning Project

## Data Download and Setup

Download the dataset from [Kaggle](https://www.kaggle.com/datasets/gpreda/chinese-mnist).

## Environment Setup

To set up the environment, run the following commands in your Jupyter Notebook:

```python
!pip install torch torchvision pandas matplotlib
!wget -O chinese_mnist.csv https://www.kaggle.com/datasets/gpreda/chinese-mnist/download

# Unzip the dataset if necessary
# !unzip path/to/your/downloaded/zipfile.zip -d target_directory
