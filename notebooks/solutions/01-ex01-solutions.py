y.value_counts()

log_reg.fit(X_train, y_train)

log_reg.score(X_test, y_test)

dummy_scores = cross_val_score(dummy_clf, X_train, y_train, scoring="roc_auc")
dummy_scores.mean()

knc_scores = cross_val_score(knc, X_train, y_train, scoring="roc_auc")
knc_scores.mean()

log_reg_scores = cross_val_score(log_reg, X_train, y_train, scoring="roc_auc")
log_reg_scores.mean()
