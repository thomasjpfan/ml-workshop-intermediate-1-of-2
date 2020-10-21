log_reg.fit(X_train, y_train)
log_reg.score(X_test, y_test)

dummy_scores = cross_val_score(dummy_clf, X_train, y_train, scoring='roc_auc_ovr')
dummy_scores.mean()

knc_scores = cross_val_score(knc, X_train, y_train, scoring="roc_auc_ovr")
knc_scores.mean()

log_reg = make_pipeline(StandardScaler(), LogisticRegression(random_state=0))

log_reg_scores = cross_val_score(log_reg, X_train, y_train, scoring="roc_auc_ovr")
log_reg_scores.mean()
