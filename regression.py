# regression.py

import utils
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor

def main():
    """Main function to run the regression model comparison."""
    df = utils.load_data()
    X_train, X_test, y_train, y_test = utils.split_data(df)

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
    }

    print("--- Model Performance Comparison ---")
    for name, model in models.items():
        trained_model = utils.train_model(model, X_train, y_train)
        mse, r2 = utils.evaluate_model(trained_model, X_test, y_test)
        print(f"\nModel: {name}")
        print(f"  MSE: {mse:.4f}")
        print(f"  R2 Score: {r2:.4f}")

if __name__ == "__main__":
    main()