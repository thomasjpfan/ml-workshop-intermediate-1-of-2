from sklearn.model_selection import cross_validate
import pandas as pd

results = cross_validate(log_reg, X_train, y_train, cv=4)

pd.DataFrame(results)

multi_results = cross_validate(log_reg, X_train, y_train, cv=4, scoring=["f1", "accuracy"])

pd.DataFrame(multi_results)
