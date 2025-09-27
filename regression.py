# regression.py (for hyper_branch)

import utils
from sklearn.ensemble import RandomForestRegressor

def main():
    """Main function to run hyperparameter tuning for RandomForest."""
    df = utils.load_data()
    X_train, X_test, y_train, y_test = utils.split_data(df)

    # The hyperparameter values we want to test
    n_estimator_values = [50, 100, 150]

    print("--- Hyperparameter Tuning for RandomForestRegressor ---")
    for n in n_estimator_values:
        model = RandomForestRegressor(n_estimators=n, random_state=42)
        trained_model = utils.train_model(model, X_train, y_train)
        mse, r2 = utils.evaluate_model(trained_model, X_test, y_test)

        print(f"\nHyperparameter: n_estimators = {n}")
        print(f"  MSE: {mse:.4f}")
        print(f"  R2 Score: {r2:.4f}")

if __name__ == "__main__":
    main()