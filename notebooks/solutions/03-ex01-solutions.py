
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

cancer = fetch_openml(data_id=15, as_frame=True)

print(cancer.DESCR)

X, y = cancer.data, cancer.target

X.shape

X.isna().sum()

imputer = SimpleImputer(add_indicator=True)
X_trans = imputer.fit_transform(X)

X_trans.shape

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, stratify=y
)

log_reg = make_pipeline(SimpleImputer(add_indicator=True), StandardScaler(), LogisticRegression())

log_reg.fit(X_train, y_train)

log_reg.score(X_test, y_test)
