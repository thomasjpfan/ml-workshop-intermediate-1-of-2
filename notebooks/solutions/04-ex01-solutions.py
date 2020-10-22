ames = fetch_openml(data_id=41211, as_frame=True)

X, y = ames.data, ames.target

X.shape

y.iloc[:10]

categorical_features = X.select_dtypes(include='category').columns

numerical_features = X.select_dtypes(include='number').columns

len(categorical_features)

len(numerical_features)

cat_prep = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='sk_missing')),
    ('ohe', OneHotEncoder(handle_unknown='ignore', sparse=False))
])

ct = ColumnTransformer([
    ('numerical', 'passthrough', numerical_features),
    ('categorical', cat_prep, categorical_features)
])

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42)

from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor

hist = Pipeline([
    ('prep', ct),
    ('estimator', HistGradientBoostingRegressor(random_state=42))
])
hist

%%time
hist.fit(X_train, y_train)

hist.score(X_train, y_train)

hist.score(X_test, y_test)
