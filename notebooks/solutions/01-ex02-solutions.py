from sklearn.model_selection import cross_validate
import pandas as pd

metrics = cross_validate(log_reg, X_train, y_train, cv=4)

pd.DataFrame(metrics)

extra_metrics = cross_validate(log_reg, X_train, y_train, cv=4, scoring=['f1_macro', 'accuracy'])

pd.DataFrame(extra_metrics)
