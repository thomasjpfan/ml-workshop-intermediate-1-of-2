
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_dist = {
    "max_features": randint(1, 11),
    "min_samples_split": randint(2, 11)
}

random_search_rf = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_jobs=8, verbose=1, random_state=42
)

random_search_rf.fit(X_train, y_train)

random_search_rf.score(X_test, y_test)

from sklearn.svm import SVC

random_search = GridSearchCV(
    SVC(random_state=42), param_grid={'kernel': ['linear', 'poly', 'rbf', 'sigmoid']},
    verbose=1, n_jobs=8
)

random_search.fit(X_train, y_train)

random_search.best_score_

random_search.score(X_test, y_test)

random_search.best_params_
