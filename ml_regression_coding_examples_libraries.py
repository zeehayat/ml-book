"""
Beginner ML Regression with Python libraries.

Uses NumPy, pandas, scikit-learn, matplotlib, and optional statsmodels.
Run:
    python3 ml_regression_coding_examples_libraries.py
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="matplotlib-"))


def make_house_dataframe(n: int = 180, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    size_sqft = rng.uniform(650, 3200, size=n)
    bedrooms = rng.choice([1, 2, 3, 4, 5], size=n)
    age_years = rng.uniform(0, 45, size=n)
    region = rng.choice(["North", "South", "Central"], size=n, p=[0.35, 0.30, 0.35])

    region_bonus = np.select(
        [region == "North", region == "South", region == "Central"],
        [28.0, -12.0, 0.0],
    )
    noise = rng.normal(0, 20, size=n)
    price_k = 45 + 0.16 * size_sqft + 18 * bedrooms - 1.2 * age_years + region_bonus + noise

    return pd.DataFrame(
        {
            "size_sqft": size_sqft,
            "bedrooms": bedrooms,
            "age_years": age_years,
            "region": region,
            "price_k": price_k,
        }
    )


def main() -> None:
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import KFold, cross_validate, train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

    df = make_house_dataframe()
    print("\nData preview")
    print(df.head())
    print("\nMissing values")
    print(df.isna().sum())

    X = df[["size_sqft", "bedrooms", "age_years", "region"]]
    y = df["price_k"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
    )

    numeric_features = ["size_sqft", "bedrooms", "age_years"]
    categorical_features = ["region"]

    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first")),
        ]
    )
    preprocess = ColumnTransformer(
        [
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )

    models = {
        "LinearRegression": LinearRegression(),
        "RidgeCV": RidgeCV(alphas=np.logspace(-3, 3, 40)),
        "LassoCV": LassoCV(alphas=np.logspace(-3, 1, 40), max_iter=20000, random_state=42),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=250,
            random_state=42,
            min_samples_leaf=3,
        ),
    }

    print("\nTrain/test evaluation")
    for name, estimator in models.items():
        pipeline = Pipeline([("preprocess", preprocess), ("model", estimator)])
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)
        rmse = mean_squared_error(y_test, pred) ** 0.5
        print(
            f"{name:22s} "
            f"MAE={mean_absolute_error(y_test, pred):7.2f} "
            f"RMSE={rmse:7.2f} "
            f"R2={r2_score(y_test, pred):6.3f}"
        )

    # Cross-validation with a leakage-safe pipeline.
    ridge_pipeline = Pipeline([("preprocess", preprocess), ("model", RidgeCV(alphas=np.logspace(-3, 3, 40)))])
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    results = cross_validate(
        ridge_pipeline,
        X,
        y,
        cv=cv,
        scoring={
            "mae": "neg_mean_absolute_error",
            "rmse": "neg_root_mean_squared_error",
            "r2": "r2",
        },
    )
    print("\n5-fold cross-validation for Ridge pipeline")
    print("MAE :", round(-results["test_mae"].mean(), 2))
    print("RMSE:", round(-results["test_rmse"].mean(), 2))
    print("R2  :", round(results["test_r2"].mean(), 3))

    # Polynomial regression pipeline.
    numeric_poly = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("scaler", StandardScaler()),
        ]
    )
    poly_preprocess = ColumnTransformer(
        [
            ("num_poly", numeric_poly, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )
    poly_pipeline = Pipeline(
        [
            ("preprocess", poly_preprocess),
            ("model", RidgeCV(alphas=np.logspace(-3, 3, 40))),
        ]
    )
    poly_pipeline.fit(X_train, y_train)
    poly_pred = poly_pipeline.predict(X_test)
    print("\nPolynomial Ridge pipeline")
    print("MAE :", round(mean_absolute_error(y_test, poly_pred), 2))
    print("RMSE:", round(mean_squared_error(y_test, poly_pred) ** 0.5, 2))
    print("R2  :", round(r2_score(y_test, poly_pred), 3))

    # Optional inference-style report. This is useful, but not required for ML prediction.
    try:
        import statsmodels.formula.api as smf

        stats_model = smf.ols(
            "price_k ~ size_sqft + bedrooms + age_years + C(region)",
            data=df,
        ).fit()
        print("\nstatsmodels coefficient table")
        print(stats_model.summary().tables[1])
    except Exception as exc:
        print("\nstatsmodels section skipped:", exc)

    # Optional plot file.
    try:
        import matplotlib.pyplot as plt

        final_model = Pipeline([("preprocess", preprocess), ("model", RidgeCV(alphas=np.logspace(-3, 3, 40)))])
        final_model.fit(X_train, y_train)
        pred = final_model.predict(X_test)
        plt.figure(figsize=(6, 5))
        plt.scatter(y_test, pred, alpha=0.75)
        low = min(y_test.min(), pred.min())
        high = max(y_test.max(), pred.max())
        plt.plot([low, high], [low, high], linestyle="--")
        plt.xlabel("Actual price_k")
        plt.ylabel("Predicted price_k")
        plt.title("Regression: Actual vs Predicted")
        plt.tight_layout()
        plt.savefig("ml_regression_actual_vs_predicted.png", dpi=160)
        print("\nSaved plot: ml_regression_actual_vs_predicted.png")
    except Exception as exc:
        print("\nPlot section skipped:", exc)


if __name__ == "__main__":
    main()
