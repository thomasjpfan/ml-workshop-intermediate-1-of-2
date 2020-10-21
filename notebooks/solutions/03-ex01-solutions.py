from sklearn.datasets import fetch_openml

cancer = fetch_openml(data_id=15, as_frame=True)

print(cancer.DESCR)

X, y = cancer.data, cancer.target

X.isnull().sum()

X.shape

imputer = SimpleImputer(add_indicator=True)
X_trans = imputer.fit_transform(X)

X_trans.shape

# Check that the 1's correspond to rows with a missing Bare_Nuclei
X.loc[X['Bare_Nuclei'].isna()].index

np.flatnonzero(X_trans[:, -1] == 1)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, stratify=y
)

pipe = Pipeline([
    ('impute', SimpleImputer(add_indicator=True)),
    ('scale', StandardScaler()),
    ('log_reg', LogisticRegression())
])

pipe.fit(X_train, y_train)

pipe.score(X_test, y_test)
