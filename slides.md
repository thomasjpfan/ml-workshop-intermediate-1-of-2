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

# Mini Review Of Scikit-learn API

- estimators
- regressors/classifiers
- transformers

---

name: validation
class: chapter-slide

# 1. Cross Validation

.footnote[
[Back to Table of Contents](#table-of-contents)
]

???

- Show train test split
- Show cross validation `cross_val_score`
- Assuming iid
- `KFold`
- `StratifiedKFold`
- Why Straifed?
- `LeaveOneOut`
- `ShuffleSplit`
- `RepeatedKFold`
- Grouped data
- Time series
- notebook

---

name: parameter-tuning
class: chapter-slide

# 2. Parameter Tuning

.footnote[
[Back to Table of Contents](#table-of-contents)
]

???

- Model complexity
- Over fitting and underfitting
- Can overfit the validation set
- notebook

---

name: pandas
class: chapter-slide

# 3. Pandas Interoperability

.footnote[
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

.footnote[
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
