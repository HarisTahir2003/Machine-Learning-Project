# Hi, I'm Haris! 👋

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/) 

# Machine Learning Project

The repository contains the following files:

* **Project Document.pdf** – contains detailed instructions and guidlines related to the project <br>

* **scraping.ipynb** – contains code for scraping data from prominent Urdu news websites
* **preprocessing.ipynb** – contains code for preprocessing the raw scraped textual data
* **raw_data.csv** – a csv file containing the actual raw scraped data
* **scraped_data.csv** – a csv file containing the preprocessed data <br>
  
* **NaiveBayes.ipynb** – trains a Multinomial Naive Bayes model on the data
* **LogisticRegression.ipynb** – trains a Multiclass Logistic Regression model on the data
* **NeuralNetwork.ipynb** – trains a simple feedforward Neural Network classifier on the data <br>
* **Report.pdf** - a detailed report that discusses the methodology and results of the project


## Table of Contents

1. [Introduction](#introduction)
2. [Installation Requirements](#installation-requirements)
3. [Project Overview](#project-overview)
4. [Data](#data)
5. [Training and Evaluation](#training-and-visualization)
6. [Screenshots](#screenshots)
   
## Introduction

This machine learning project involves building a text classification system for Urdu news articles. The pipeline includes web scraping, data preprocessing, and training multiple classifiers: Naive Bayes, Logistic Regression, and a simple Neural Network.

**Classification Models Used**:
* **Naive Bayes Classifier**: Naive Bayes is a probabilistic classifier based on Bayes' Theorem. It assumes that features are conditionally independent given the class label. Despite this "naive" assumption, it performs well on text classification tasks due to the natural independence of words.

* **Logistic Regression**: Logistic Regression is a linear model used for classification that estimates the probability of a class using a sigmoid or softmax function. In this project, we use a multiclass version, which can handle more than two categories.

* **Neural Network Classifier**: A simple feedforward neural network (also known as a Multi-Layer Perceptron or MLP) is implemented, consisting of fully connected layers. Neural networks can capture non-linear relationships and are well-suited for more complex text features.

**What is Scraping?**
Web scraping is the process of automatically extracting data from websites. In this project, we scraped Urdu news articles from several prominent sources. This raw data served as the foundation for training and evaluating the machine learning models.


## Installation Requirements

To run the notebooks in this repository, you will need the following packages:

* `numpy`
* `pandas`
* `matplotlib`
* `scikit-learn`
* `seaborn`
* `regex`
* `keras`
* `tensorflow`
* `LughaatNLP`
* `torch`
* `os`
* `json`
* `time`
* `random`
* `zipfile`
* `requests`
* `BeautifulSoup`

You can install these packages using pip:

```bash
pip install numpy
```

```bash
pip install pandas
```

```bash
pip install matplotlib
```

```bash
pip install scikit-learn
```

```bash
pip install seaborn
```

```bash
pip install regex
```

```bash
pip install keras
```

```bash
pip install tensorflow
```

```bash
pip install LughaatNLP
```

```bash
pip install torch
```

```bash
pip install requests
```

The os, json, time, random and zipfile libraries are part of Python's standard library and do not need to be installed via pip. <br>

To install the BeautifulSoup library, simply run the first code block in the scraping.ipynb file. 

```bash
!pip install BeautifulSoup4
```

After installing the required libraries, run the **"Imports"** cell in each notebook to begin.

Useful Links for installing Jupyter Notebook:
- https://youtube.com/watch?v=K0B2P1Zpdqs  (MacOS)
- https://www.youtube.com/watch?v=9V7AoX0TvSM (Windows)

It's recommended to run this notebook in a conda environment to avoid dependency conflicts and to ensure smooth execution.
Also, you will need a GPU to run the notebooks. It is recommended to have a Google Colab Account (perhaps multiple accounts) for this purpose.
<h4> Conda Environment Setup </h4>
<ul> 
   <li> Install conda </li>
   <li> Open a terminal/command prompt window in the assignment folder. </li>
   <li> Run the following command to create an isolated conda environment titled AI_env with the required packages installed: conda env create -f environment.yml </li>
   <li> Open or restart your Jupyter Notebook server or VSCode to select this environment as the kernel for your notebook. </li>
   <li> Verify the installation by running: conda list -n AI_env </li>
   <li> Install conda </li>
</ul>


## Project Overview

Our project was executed over a period of approximately one month. The goal was to classify Urdu news articles into five distinct categories using supervised machine learning techniques. We implemented and evaluated three classifiers: **Multinomial Naive Bayes**, **Multiclass Logistic Regression**, and a **Neural Network**, analyzing their performance on real-world scraped news data.


### 1. **Dataset Collection**

We scraped 2,750 Urdu news articles from major Pakistani news websites including *Geo Urdu*, *Jang*, *Dunya News*, *Express News*, and *Samaa News*. These articles were categorized into five classes: **Entertainment**, **Business**, **Sports**, **Science-Technology**, and **International**. The scraping code is available in `scraping.ipynb`, and the collected articles were saved in `raw_data.csv`.

### 2. **Data Preprocessing**

The raw data was cleaned and transformed using custom techniques and the `LughaatNLP` library. This included removing duplicates, handling missing data, correcting spelling errors, and applying normalization, lemmatization, and stemming. Stopwords were removed and the text was tokenized to prepare it for modeling. The final preprocessed dataset was saved as `scraped_data.csv`, and all preprocessing steps are documented in `preprocessing.ipynb`.

### 3. **Model 1: Multinomial Naive Bayes**

We began with a **Multinomial Naive Bayes** classifier using the Bag-of-Words representation. The model achieved an accuracy of **96.55%** and a macro F1-score of **0.97**, demonstrating strong performance across all five categories. Implementation details are in `NaiveBayes.ipynb`.

### 4. **Model 2: Logistic Regression**

Our second model used **Multiclass Logistic Regression** with a Softmax activation function. This model achieved **95.27%** accuracy and a macro F1-score of **0.95**. It demonstrated robust results, particularly for the **Business** and **Sports** categories. The code for this model is available in `LogisticRegression.ipynb`.

### 5. **Model 3: Neural Network**

Finally, we implemented a simple **feedforward neural network** using Pytorch. This model used tokenized sequences as input and included techniques like batch normalization and dropout. It achieved the **highest accuracy of 97.45%** and a macro F1-score of **0.97**. The implementation is in `NeuralNetwork.ipynb`.

### 6. **Report**

Our findings, methodologies, and evaluations are comprehensively summarized in `Report.pdf`. This document outlines our motivations, preprocessing strategies, model architectures, and performance insights. It also discusses limitations such as single-label classification and the lack of contextual understanding in traditional models.


## Data

The dataset for this project was curated by scraping news articles from five major Urdu news websites in Pakistan: <br>
* **Geo Urdu**
* **Jang**
* **Dunya News**
* **Express News**
* **Samaa News**. <br>

A total of **2,750 articles** were collected using automated scraping techniques, documented in `scraping.ipynb`. These articles were retrieved directly from the websites' HTML content, extracting the article title, body text, and category labels where available. The raw data was stored in `raw_data.csv`, containing multiple fields including article text, publication date, source URL, and category. The articles were labeled into one of five predefined categories: <br>
* **Entertainment**
* **Business**
* **Sports**
* **Science-Technology**
* **International**.

After collection, the data underwent a comprehensive cleaning and preprocessing phase to remove noise such as HTML tags, duplicates, and non-Urdu content. Additionally, tokenization, stopword removal, normalization, stemming, and lemmatization were applied using the `LughaatNLP` library, resulting in a final cleaned dataset saved in `scraped_data.csv`. This dataset forms the foundation for training and evaluating the three supervised learning models implemented in the project.


## Training and Visualization

The entire training process alongside the relevant evaluations and visualizations are explained in detail in the jupyter notebook. 


## Screenshots

<h4> 1. This image shows eight random samples of Chest X-ray images from the Image Segmentation dataset, with their masks ovelaid on top of the grayscale images. The masks are shaded red for better visualization.  </h4>
<img src="Pic1.png" width="450px"> <br> 

<h4> 2. This image shows the confusion matrix when the SEResNet50 classifier is evaluated on the test dataset containing raw Chest X-ray images. </h4>
<img src="Pic2.png" width="450px"> <br> 

<h3> 3. Swin Transformer </h3>
<h4> This image shows the confusion matrix when the Swin Transformer model is evaluated on the test dataset. </h4>
<img src="SwinTransformer_confusion_matrix.png" width="450px">
 <br> 
 
## License

[MIT](https://choosealicense.com/licenses/mit/)

