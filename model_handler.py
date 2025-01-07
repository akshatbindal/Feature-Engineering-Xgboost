from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

def prepare_data(data, selected_features, target_feature):
    features = data[selected_features + [target_feature]].dropna()
    X = features.drop(target_feature, axis=1)
    y = features[target_feature]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler

def train_model(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        alpha=0.2,
        random_state=42
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    return model, (X_train, y_train), (X_val, y_val), (X_test, y_test)

def evaluate_model(model, X, y):
    predictions = model.predict(X)
    metrics = {
        "MSE": mean_squared_error(y, predictions),
        "MAE": mean_absolute_error(y, predictions),
        "R2": r2_score(y, predictions),
        "MAPE": mean_absolute_percentage_error(y, predictions)
    }
    return predictions, metrics
