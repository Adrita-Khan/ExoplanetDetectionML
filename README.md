# Exoplanet ML

![Exoplanet Image](/Images/Hr8799_orbit_hd.gif)

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![GitHub issues](https://img.shields.io/github/issues/yourusername/exoplanet-ml)
![GitHub stars](https://img.shields.io/github/stars/yourusername/exoplanet-ml?style=social)

**Exoplanet ML** is a machine learning project dedicated to the detection of exoplanets using transit survey-based light curves. By leveraging advanced machine learning algorithms and feature engineering techniques, this project aims to enhance the accuracy and efficiency of exoplanet discovery.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Examples](#examples)
- [Machine Learning Algorithms](#machine-learning-algorithms)
- [Key Notebooks](#key-notebooks)
- [Project Structure](#project-structure)
- [Resources](#resources)
  - [Dimensionality Reduction](#dimensionality-reduction)
  - [Feature Selection](#tsfresh-feature-selection)
  - [Supervised Learning](#scikit-learn-supervised-learning-list-and-description)
  - [Gaussian Process](#gaussian-process)
  - [Unsupervised Learning](#scikit-learn-unsupervised-learning-list-and-description)
  - [Hyperparameter Tuning](#hyperopt-hyperparameter-tuning)
  - [Incremental PCA](#incremental-principal-component-analysis)
  - [Plotting](#scikit-learn-plotting)
  - [Probability Calibration](#probability-calibration)
  - [Additional Resources](#technical-problem-solution-and-miscellaneous-links)
- [Results](#results)
- [Contributing](#contributing)
- [Acknowledgements](#acknowledgements)
- [License](#license)
- [Contact](#contact)

## Introduction

The search for exoplanets has been revolutionized by space-based transit surveys, which monitor the brightness of stars to detect periodic dips caused by orbiting planets. **Exoplanet ML** harnesses the power of machine learning to analyze these light curves, improving the detection rate and reducing false positives. This project integrates various machine learning techniques, from feature engineering to model calibration, to provide a comprehensive toolkit for astronomers and data scientists alike.

## Features

- **Automated Exoplanet Detection**: Utilizes transit survey-based light curves to identify potential exoplanets.
- **Advanced Algorithms**: Implements state-of-the-art machine learning models for high accuracy.
- **Feature Engineering**: Employs robust feature extraction and selection techniques to enhance model performance.
- **Dimensionality Reduction**: Reduces feature space complexity while preserving essential information.
- **Scalable Pipelines**: Designed to handle large datasets efficiently.
- **Model Calibration**: Ensures probability estimates are well-calibrated for reliable predictions.

## Installation

To get started with **Exoplanet ML**, follow these steps:

1. **Clone the repository:**

    ```bash
    git clone https://github.com/yourusername/exoplanet-ml.git
    cd exoplanet-ml
    ```

2. **Create a virtual environment:**

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## Usage

Here's a quick example of how to use **Exoplanet ML** to train a model and make predictions:

1. **Prepare your data:**

    Ensure your light curve data is in the correct format as specified in the [Data Preparation Guide](docs/data_preparation.md).

2. **Run the training script:**

    ```bash
    python train.py --config config.yaml
    ```

3. **Make predictions:**

    ```bash
    python predict.py --model models/best_model.pkl --input data/new_lightcurves.csv
    ```

For detailed instructions, refer to the [Usage Guide](docs/usage.md).

## Examples

Below are some examples of model performance and visualizations:

### Model Performance

| Model                     | Accuracy | F1 Score | AUC-ROC |
|---------------------------|----------|----------|---------|
| Random Forest Classifier  | 95%      | 0.94     | 0.98    |
| LightGBM                  | 96%      | 0.95     | 0.99    |
| AdaBoost                  | 93%      | 0.92     | 0.97    |
| XGBoost                   | 96%      | 0.95     | 0.99    |

### Feature Importance

![Feature Importance](Images/feature_importance.png)

### Light Curve Visualization

![Light Curve](Images/light_curve_example.png)

## Machine Learning Algorithms

Exoplanet ML employs a variety of machine learning algorithms to ensure comprehensive analysis and accurate predictions:

- **Random Forest Classifier**
- **LightGBM**
- **AdaBoost**
- **Histogram Gradient Boosting**
- **XGBoost**
- **XGBoost Calibrated**

## Key Notebooks

- [Kepler Lightcurve Notebook](https://spacetelescope.github.io/notebooks/notebooks/MAST/Kepler/Kepler_Lightcurve/kepler_lightcurve.html)
- [Feature Engineering with TSFresh](notebooks/feature_engineering_tsfresh.ipynb)
- [Model Training and Evaluation](notebooks/model_training_evaluation.ipynb)

## Project Structure

plaintext
exoplanet-ml/
├── data/
│   ├── raw/
│   ├── processed/
│   └── new_lightcurves.csv
├── notebooks/
│   ├── feature_engineering_tsfresh.ipynb
│   ├── model_training_evaluation.ipynb
│   └── exploratory_analysis.ipynb
├── models/
│   ├── best_model.pkl
│   └── model_v1.pkl
├── src/
│   ├── data_preparation.py
│   ├── feature_engineering.py
│   ├── train.py
│   └── predict.py
├── docs/
│   ├── data_preparation.md
│   └── usage.md
├── Images/
│   ├── Hr8799_orbit_hd.gif
│   ├── feature_importance.png
│   └── light_curve_example.png
├── tests/
│   ├── test_data_preparation.py
│   └── test_model.py
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
