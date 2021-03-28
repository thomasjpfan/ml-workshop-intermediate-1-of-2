ames = fetch_openml(data_id=41211, as_frame=True)

X, y = ames.data, ames.target

X.shape

cat_cols = X.select_dtypes(include='category').columns

num_cols = X.select_dtypes(include='number').columns

cat_cols.shape

num_cols.shape

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=0)

from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor

preprocess = ColumnTransformer([
    ('categorical', OneHotEncoder(handle_unknown='ignore', sparse=False), cat_cols),
    ('numerical', 'passthrough', num_cols)
])

hist = Pipeline([
    ('preprocess', preprocess),
    ('regressor', HistGradientBoostingRegressor(random_state=0))
])

hist.fit(X_train, y_train)

hist.score(X_test, y_test)
