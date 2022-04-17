from scipy.stats import randint

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV

param_dist = {
    "max_features": randint(1, 11),
    "min_samples_split": randint(2, 11)
}

search_cv = RandomizedSearchCV(RandomForestClassifier(random_state=0),
                               param_distributions=param_dist, n_iter=20, verbose=1, n_jobs=8, random_state=0)

search_cv.fit(X_train, y_train)

search_cv.best_params_

search_cv.best_score_

search_cv.score(X_test, y_test)

half_cv = HalvingRandomSearchCV(RandomForestClassifier(random_state=0),
                                param_distributions=param_dist, verbose=1, n_jobs=8, random_state=0)

half_cv.fit(X_train, y_train)

half_cv.best_params_

half_cv.best_score_

half_cv.score(X_test, y_test)
