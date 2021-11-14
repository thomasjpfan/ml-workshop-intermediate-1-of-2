from scipy.stats import randint

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV

param_dist = {
    "max_features": randint(1, 11),
    "min_samples_split": randint(2, 11)
}

random_search_cv = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=20,
    verbose=1,
    n_jobs=2,
)

random_search_cv.fit(X_train, y_train)

random_search_cv.best_params_

random_search_cv.score(X_test, y_test)

rsh = HalvingRandomSearchCV(
    estimator=RandomForestClassifier(random_state=42), param_distributions=param_dist, random_state=42,
    n_jobs=2, verbose=1
)

rsh.fit(X_train, y_train)

rsh.score(X_test, y_test)
