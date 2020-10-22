title: Intermediate Machine Learning with scikit-learn: Cross validation, Parameter Tuning, Pandas Interoperability, and Missing Values
use_katex: True
class: title-slide

# Intermediate Machine Learning with scikit-learn
## Cross validation, Parameter Tuning, Pandas Interoperability, and Missing Values

![](images/scikit-learn-logo-notext.png)

.larger[Thomas J. Fan]<br>
@thomasjpfan<br>
<a href="https://www.github.com/thomasjpfan" target="_blank"><span class="icon icon-github icon-left"></span></a>
<a href="https://www.twitter.com/thomasjpfan" target="_blank"><span class="icon icon-twitter"></span></a>
<a class="this-talk-link", href="https://github.com/thomasjpfan/ml-workshop-intermediate-1-of-2" target="_blank">
This workshop on Github: github.com/thomasjpfan/ml-workshop-intermediate-1-of-2</a>

---

name: table-of-contents
class: title-slide, left

# Table of Contents
.g[
.g-6[
1. [Cross Validation](#validation)
1. [Parameter Tuning](#parameter-tuning)
1. [Missing Values](#missing-values)
1. [Pandas Interoperability](#pandas)
]
.g-6.g-center[
![](images/scikit-learn-logo-notext.png)
]
]

---

# Scikit-learn API

.center[
## `estimator.fit(X, [y])`
]

.g[
.g-6[
## `estimator.predict`
- Classification
- Regression
- Clustering
]
.g-6[
## `estimator.transform`
- Preprocessing
- Dimensionality reduction
- Feature selection
- Feature extraction
]
]

---

# Data Representation

![:scale 80%](images/data-representation.svg)

---

# Supervised ML Workflow

![](images/ml-workflow-sklearn.svg)

---

class: chapter-slide

# Notebook 📒!
## notebooks/00-review-sklearn.ipynb

---

name: validation
class: chapter-slide

# 1. Cross Validation

.footnote-back[
[Back to Table of Contents](#table-of-contents)
]

---

# Single train test split

![:scale 70%](images/train-test.svg)

---

# Three Fold Split

![:scale 75%](images/split-data-three.svg)

---

# Why cross validate?

![:scale 80%](notebooks/images/overfitting_validation_set_1.svg)

---

# Why cross validate?

![:scale 80%](notebooks/images/overfitting_validation_set_2.svg)

---

# Can we do better?

![:scale 80%](images/grid_search_cross_validation.png)

---

class: chapter-slide

# Notebook 📓!
## notebooks/01-cross-validation.ipynb

---

class: chapter-slide

# Cross Validation Strategies

---

![:scale 90%](images/kfold_cv.png)

---

![:scale 90%](images/stratified_cv.png)

---

![:scale 90%](images/shuffle_split_cv.png)

---

![:scale 90%](images/repeated_stratified_kfold.png)

---

# Strategies for increasing the number of folds

- High variance, takes a long time
```py
from sklearn.model_selection import LeaveOneOut
```

- `ShuffleSplit` with stratification
```py
from sklearn.model_selection import StratifiedShuffleSplit
```

- Repeat `KFold` or `StratifiedKFold`
```py
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import RepeatedStratifiedKFold
```

---

# Cross-validation with non-idd data

## Grouped data

- Assume data is not iid such as patient ID or user id
- We want to generalized to a new patient

## Time Series

- Data is correlated

---

![:scale 90%](images/group_kfold.png)

---

![:scale 90%](notebooks/images/approval_ratings.png)

---

![:scale 90%](notebooks/images/approval_ratings_random.png)

---

![:scale 90%](notebooks/images/approval_ratings_structured.png)

---

![:scale 90%](images/time_series_cv.png)

---

![:scale 90%](images/time_series_walk_forward_cv.png)

---

class: chapter-slide

# Notebook 📓!
## notebooks/01-cross-validation.ipynb

---

name: parameter-tuning
class: chapter-slide

# 2. Parameter Tuning

.footnote-back[
[Back to Table of Contents](#table-of-contents)
]

---

class: center

# What Tune Parameters?

![:scale 50%](notebooks/images/knn_boundary_n_neighbors.png)

---

# Score vs n_neighbors

![:scale 80%](notebooks/images/knn_model_complexity.png)

---

# Parameter Tuning Workflow

![:scale 80%](images/gridsearch_workflow.png)

---

# GridSearchCV

```py
from sklearn.model_selection import GridSearchCV

param_grid = {'n_neighbors': np.arange(1, 30, 2)}
grid = GridSearchCV(KNeighborsClassifier(), param_grid=param_grid,
                    return_train_score=True)

grid.fit(X_train, y_train)
```

Best score

```py
grid.best_score_
```

Best parameters

```py
grid.best_params_
```

---

# Random Search

![](images/bergstra_random.jpeg)

---

# RandomizedSearchCV with scikit-learn

```py
from scipy.stats import randint
param_dist = {
    "max_depth": [3, None],
    "max_features": randin(1, 11)
}

random_search = RandomizedSearchCV(
    clf,
    param_distributions=param_dist,
    n_iter=200
)
```

- Values in `param_distributions` can be a list or an object from the
`scipy.stats` module

---

class: chapter-slide

# Notebook 📓!
## notebooks/02-parameter-tuning.ipynb

---

name: missing-values
class: chapter-slide

# 3. Missing Values

.footnote-back[
[Back to Table of Contents](#table-of-contents)
]

---

# Imputers in scikit-learn

## Impute module

```py
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer

# `add_indicator=True` to add missing indicator
imputer = SimpleImputer(add_indicator=True)

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
```

---

# Comparing the Different methods

![:scale 100%](images/med_knn_rf_comparison.png)

---

# Estimators with native support

## Histogram-based Gradient Boosting Regression Trees

- Based on LightGBM implementation
- Have native support for missing values

```py
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingRegressor
```

---

class: chapter-slide

# Notebook 📔!
## notebooks/03-missing-values.ipynb

---

name: pandas
class: chapter-slide

# 4. Pandas Interoperability

.footnote-back[
[Back to Table of Contents](#table-of-contents)
]

---

# Categorical Data

## Examples of categories:

- `['Manhattan', 'Queens', 'Brooklyn', 'Bronx']`
- `['dog', 'cat', 'mouse']`

## Scikit-learn Encoders

```py
from sklearn.preprocessing import OrdinalEncoder
```
- Encodes categories into an integer

```py
from sklearn.preprocessing import OneHotEncoder
```
- Binary column for each category

---

# Heterogenous data

## Example: Titanic Dataset

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pclass</th>
      <th>sex</th>
      <th>age</th>
      <th>sibsp</th>
      <th>parch</th>
      <th>fare</th>
      <th>embarked</th>
      <th>body</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>female</td>
      <td>29.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>211.3375</td>
      <td>S</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>male</td>
      <td>0.9167</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>151.5500</td>
      <td>S</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>female</td>
      <td>2.0000</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>151.5500</td>
      <td>S</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>male</td>
      <td>30.0000</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>151.5500</td>
      <td>S</td>
      <td>135.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.0</td>
      <td>female</td>
      <td>25.0000</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>151.5500</td>
      <td>S</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>

---

# scikit-learn's ColumnTransformer

![:scale 100%](images/column_transformer_schematic.png)

---

class: chapter-slide

# Notebook 📔!
## notebooks/04-pandas-interoperability.ipynb

---

class: title-slide, left

# Closing

.g.g-middle[
.g-7[
![:scale 30%](images/scikit-learn-logo-notext.png)
1. [Cross Validation](#validation)
1. [Parameter Tuning](#parameter-tuning)
1. [Missing Values](#missing-values)
1. [Pandas Interoperability](#pandas)
]
.g-5.center[
<br>
.larger[Thomas J. Fan]<br>
@thomasjpfan<br>
<a href="https://www.github.com/thomasjpfan" target="_blank"><span class="icon icon-github icon-left"></span></a>
<a href="https://www.twitter.com/thomasjpfan" target="_blank"><span class="icon icon-twitter"></span></a>
<a class="this-talk-link", href="https://github.com/thomasjpfan/ml-workshop-intermediate-1-of-2" target="_blank">
This workshop on Github: github.com/thomasjpfan/ml-workshop-intermediate-1-of-2</a>
]
]
