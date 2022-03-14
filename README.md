# Intermediate Machine Learning with scikit-learn
### Cross validation, Parameter Tuning, Pandas Interoperability, and Missing Values

*By Thomas J. Fan*

[Link to slides](https://thomasjpfan.github.io/ml-workshop-intermediate-1-of-2/)

Scikit-learn is a Python machine learning library used by data science practitioners from many disciplines. We will learn about cross-validation, tuning machine learning algorithms, and pandas interoperability during this training. Cross-validation enables us to evaluate our machine learning models by splitting our data into multiple training and testing datasets. We will learn to handle missing values with imputation using univariate and multivariate techniques. Next, we will explore tuning algorithms in scikit-learn with grid search and random search. We will learn about categorical features and how to use scikit-learn's encoders to convert these categorical features into numerical features for a machine-learning algorithm to consume. Finally, we will apply the machine learning techniques on a house pricing dataset with scikit-learn's Histogram-based Gradient Boosted Trees. scikit-learn's boosted tree implementation is based on LightGBM and has similar performance characteristics.

## Obtaining the Material

### With git

The most convenient way to download the material is with git:

```bash
git clone https://github.com/thomasjpfan/ml-workshop-intermediate-1-of-2
```

Please note that I may add and improve the material until shortly before the session. You can update your copy by running:

```bash
git pull origin master
```

### Download zip

If you are not familiar with git, you can download this repository as a zip file at: [github.com/thomasjpfan/ml-workshop-intermediate-1-of-2/archive/master.zip](https://github.com/thomasjpfan/ml-workshop-intermediate-1-of-2/archive/master.zip). Please note that I may add and improve the material until shortly before the session. To update your copy please re-download the material a day before the session.

## Running the notebooks

### Local Installation

Local installation requires `conda` to be installed on your machine. The simplest way to install `conda` is to install `miniconda` by using an installer for your operating system provided at [docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html). After `conda` is installed, navigate to this repository on your local machine:

```bash
cd ml-workshop-intermediate-1-of-2
```

Then download and install the dependencies:

```bash
conda env create -f environment.yml
```

This will create a virtual environment named `ml-workshop-intermediate-1-of-2`. To activate this environment:

```bash
conda activate ml-workshop-intermediate-1-of-2
```

Finally, to start `jupyterlab` run:

```bash
jupyter lab
```

This should open a browser window with the `jupterlab` interface.

### Run with Google's Colab

If you have any issues with installing `conda` or running `jupyter` on your local computer, then you can run the notebooks on Google's Colab:

0. [Quick Review of scikit-learn](https://colab.research.google.com/github/thomasjpfan/ml-workshop-intermediate-1-of-2/blob/master/notebooks/00-review-sklearn.ipynb)
1. [Cross-Validation in scikit-learn](https://colab.research.google.com/github/thomasjpfan/ml-workshop-intermediate-1-of-2/blob/master/notebooks/01-cross-validation.ipynb)
2. [Parameter tuning](https://colab.research.google.com/github/thomasjpfan/ml-workshop-intermediate-1-of-2/blob/master/notebooks/02-parameter-tuning.ipynb)
3. [Missing values in scikit-learn](https://colab.research.google.com/github/thomasjpfan/ml-workshop-intermediate-1-of-2/blob/master/notebooks/03-missing-values.ipynb)
4. [Pandas Interoperability](https://colab.research.google.com/github/thomasjpfan/ml-workshop-intermediate-1-of-2/blob/master/notebooks/04-pandas-interoperability.ipynb)

## License

This repo is under the [MIT License](LICENSE).
