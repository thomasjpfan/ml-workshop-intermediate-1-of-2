from sklearn.datasets import fetch_openml
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.compose import make_column_selector

ames = fetch_openml(data_id=41211, as_frame=True)

X, y = ames.data, ames.target

categorical_names = X.select_dtypes(include='category').columns
numerical_names = X.select_dtypes(include='number').columns

categorical_names

numerical_names

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42
)

preprocessor = ColumnTransformer([
    ("numerical", "passthrough", numerical_names),
    ("categorical", OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_names)
])

hist = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", HistGradientBoostingRegressor(random_state=42))
])

hist.fit(X_train, y_train)

hist.score(X_test, y_test)

# Extra
num_selector = make_column_selector(dtype_include="number")
cat_selector = make_column_selector(dtype_include="category")

prep_callable = ColumnTransformer([
    ("numerical", "passthrough", num_selector),
    ("categorical", OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), cat_selector)
])

hist = Pipeline([
    ("prep", prep_callable),
    ("reg", HistGradientBoostingRegressor(random_state=42))
])

hist.fit(X_train, y_train)

hist.score(X_test, y_test)
