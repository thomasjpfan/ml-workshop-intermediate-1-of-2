
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_dist = {
    "max_features": randint(1, 11),
    "min_samples_split": randint(2, 11)
}

random_search = RandomizedSearchCV(RandomForestClassifier(random_state=0),
                                   param_distributions=param_dist,
                                   verbose=1,
                                   random_state=0)

random_search.fit(X_train, y_train)

random_search.best_params_

random_search.best_score_

random_search.score(X_test, y_test)

from sklearn.svm import SVC

svm_grid = GridSearchCV(
    SVC(random_state=42), param_grid={'kernel': ['linear', 'poly', 'rbf', 'sigmoid']},
    verbose=1, n_jobs=8
)

svm_grid.fit(X_train, y_train)

svm_grid.best_score_

svm_grid.best_params_

svm_grid.score(X_test, y_test)
