from utils.cleaning import load_ml_featureset, to_json_feature_importances

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, r2_score

def get_ml_train_test_split(test_size=0.2, random_state=42):
    df = load_ml_featureset()

    y = df["score"].astype(float)

    X = df.drop(columns=["id", "score", "scored_by", "ranked", "popularity", "members", "favorited"])
    # X = df.drop(columns=["id", "score"])
    
    X = X.fillna(0)
 
    if y.isna().any():
        y = y.fillna(y.median())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test

def train_random_forest_regressor(X_train, y_train, X_test, y_test):

    model = RandomForestRegressor(n_estimators=300, max_depth=None, n_jobs=-1, random_state=42)

    print("--- Training Random Forest Regressor ---")
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    rmse = root_mean_squared_error(y_test, preds)
    r2_score_value = r2_score(y_test, preds)

    print(f"\n--- Random Forest Results ---")
    print(f"RMSE: {rmse:.4f}")
    print(f"R^2 Score: {r2_score_value:.4f}")

    return model, preds

def get_feature_importances(model, X, top_n=50):
    importances = model.feature_importances_
    feature_names = X.columns

    sorted_pairs = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)

    my_dict = {}

    print(f"\n--- Top {top_n} Feature Importances ---")
    for name, importance in sorted_pairs:
        if len(my_dict) < top_n:
            print(f"{len(my_dict) + 1}. {name}: {importance:.3f}")
        my_dict[name] = float(importance.round(3))
        
    to_json_feature_importances(my_dict)

    return
    
def run_random_forest_feature_importance():
    X_train, X_test, y_train, y_test = get_ml_train_test_split()

    model, preds = train_random_forest_regressor(X_train, y_train, X_test, y_test)

    get_feature_importances(model, X_train, top_n=50)

    return

if __name__== "__main__":
    run_random_forest_feature_importance()

