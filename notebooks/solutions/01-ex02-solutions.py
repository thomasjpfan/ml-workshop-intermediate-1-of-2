results = cross_validate(log_reg, X_train, y_train, cv=4)

results_df = pd.DataFrame(results)

results_df

results = cross_validate(log_reg, X_train, y_train, cv=4, scoring=['f1', 'accuracy', 'roc_auc'])

results_df = pd.DataFrame(results)

results_df
