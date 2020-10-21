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
1. [Pandas Interoperability](#pandas)
1. [Missing Values](#missing-values)
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

![:scale 100%](images/kfold_cv.png)

---

![:scale 100%](images/stratified_cv.png)

---

![:scale 100%](images/shuffle_split_cv.png)

---

![:scale 100%](images/repeated_stratified_kfold.png)

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

![:scale 100%](images/group_kfold.png)

---

![:scale 100%](notebooks/images/approval_ratings.png)

---

![:scale 100%](notebooks/images/approval_ratings_random.png)

---

![:scale 100%](notebooks/images/approval_ratings_structured.png)

---

![:scale 100%](images/time_series_cv.png)

---

![:scale 100%](images/time_series_walk_forward_cv.png)

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

![](notebooks/images/knn_boundary_n_neighbors.png)

---

# Score vs n_neighbors

![](notebooks/images/knn_model_complexity.png)

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

# RandomSearchCV

![](images/bergstra_random.jpeg)

---

# Random Search with scikit-learn

```py
from scipoy.stats import randint
param_dist = {"max_depth": [3, None],
              "max_features": randin(1, 11)}

random_search = RandomizedSearchCV(
    clf, param_distributions=param_dist,
    n_iter=200)
```

---

class: chapter-slide

# Notebook 📓!
## notebooks/02-parameter-tuning.ipynb

---

name: pandas
class: chapter-slide

# 3. Pandas Interoperability

.footnote-back[
[Back to Table of Contents](#table-of-contents)
]

???

- Introduce heterogenous data
- categorical encoding
- column transformer
- notebook

---

name: missing-values
class: chapter-slide

# 4. Missing Values

.footnote-back[
[Back to Table of Contents](#table-of-contents)
]

???

- imputers
- simple ones
- knn based
- notebook

---

class: title-slide, left

# Closing

.g.g-middle[
.g-7[
![:scale 30%](images/scikit-learn-logo-notext.png)
1. [Cross Validation](#validation)
1. [Parameter Tuning](#parameter-tuning)
1. [Pandas Interoperability](#pandas)
1. [Missing Values](#missing-values)
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
