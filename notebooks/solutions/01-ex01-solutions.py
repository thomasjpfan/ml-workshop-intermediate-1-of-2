y.value_counts()

log_reg.fit(X_train, y_train)

log_reg.score(X_test, y_test)

d_auc_scores = cross_val_score(dummy_clf, X_train, y_train, scoring='roc_auc')
d_auc_scores.mean()

knc_auc_scores = cross_val_score(knc, X_train, y_train, scoring='roc_auc')
knc_auc_scores.mean()

log_reg_auc_scores = cross_val_score(log_reg, X_train, y_train, scoring='roc_auc')

log_reg_auc_scores.mean()
