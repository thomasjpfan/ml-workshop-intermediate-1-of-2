y.value_counts()

rf.fit(X_train, y_train)

rf.score(X_test, y_test)

dummy_scores = cross_val_score(dummy_clf, X_train, y_train, scoring="roc_auc")
dummy_scores.mean()

knc_scores = cross_val_score(knc, X_train, y_train, scoring="roc_auc")
knc_scores.mean()

rf_scores = cross_val_score(rf, X_train, y_train, scoring="roc_auc")
rf_scores.mean()
