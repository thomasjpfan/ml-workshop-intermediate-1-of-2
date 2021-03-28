from sklearn.model_selection import cross_validate

results = cross_validate(log_reg, X_train, y_train, cv=4)
results

import pandas as pd
pd.DataFrame(results)

more_results = cross_validate(log_reg, X_train, y_train, cv=4, scoring=["f1", "accuracy"])

pd.DataFrame(more_results)
