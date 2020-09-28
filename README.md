# Intermediate Machine Learning with scikit-learn
### Cross validation, Parameter Tuning, Pandas Interoperability, and Missing Values

*By Thomas J. Fan*

[Link to slides](https://thomasjpfan.github.io/ml-workshop-intermediate-1-of-2/)

Scikit-learn is a machine learning library in Python that is used by many data science practitioners. In this workshop, we will learn about cross validation, tuning machine learning algorithms, pandas interoperability, and handling missing values. Cross validation enables us to evaluate our machine learning models by splitting our data into multiple training and testing datasets. Hyper-parameter tuning helps us find parameter combinations that are suited for your data. Scikit-learn's ColumnTransformer is used to handle heterogenous data provided as a panda's DataFrame. Furthermore, scikit-learn provides univariate and K-Nearest Neighbor transformers for imputing missing values in our datasets.

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

## Installation

This workshop requires conda to be installed on your local machine. The easiest install conda is to install `miniconda` by using an installer for your operating system provided by [docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html). After conda is installed, navigate to this repository on your local machine:

```bash
cd ml-workshop-intermediate-1-of-2
```

To download and install the dependencies run:

```bash
conda env create -f environment.yml
```

This will install a virtual environment. To activate the environment:

```bash
conda activate ml-workshop-intermediate-1-of-2
```

Finally, to start `jupyterlab` run:

```bash
jupyter lab
```

This should open a browser window with the `jupterlab` interface.

## License

This repo is under the [MIT License](LICENSE).
