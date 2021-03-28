from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer(as_frame=True)

X, y = cancer.data, cancer.target

y.value_counts()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, stratify=y)

from sklearn.linear_model import LogisticRegression

log_reg = make_pipeline(
    StandardScaler(),
    LogisticRegression()
)

log_reg.fit(X_train, y_train)

log_reg.score(X_test, y_test)

from sklearn.metrics import f1_score
y_pred = log_reg.predict(X_test)

y_pred

f1_score(y_test, y_pred)
