"""
Beginner ML Regression in pure Python.

No NumPy, pandas, scikit-learn, or statistics packages are used.
Run:
    python3 ml_regression_coding_examples_pure_python.py
"""

from __future__ import annotations

import math
import random


def make_house_data(n: int = 80, seed: int = 7) -> tuple[list[list[float]], list[float]]:
    """Create a small synthetic house-price dataset.

    Features:
        size_sqft, bedrooms, age_years
    Target:
        price in thousands
    """
    rng = random.Random(seed)
    X: list[list[float]] = []
    y: list[float] = []
    for _ in range(n):
        size = rng.uniform(650, 3200)
        bedrooms = rng.choice([1, 2, 3, 4, 5])
        age = rng.uniform(0, 45)
        noise = rng.gauss(0, 18)
        price = 45 + 0.16 * size + 18 * bedrooms - 1.2 * age + noise
        X.append([size, float(bedrooms), age])
        y.append(price)
    return X, y


def train_test_split(
    X: list[list[float]],
    y: list[float],
    test_fraction: float = 0.25,
    seed: int = 42,
) -> tuple[list[list[float]], list[list[float]], list[float], list[float]]:
    rng = random.Random(seed)
    indices = list(range(len(X)))
    rng.shuffle(indices)
    test_size = int(round(len(X) * test_fraction))
    test_idx = set(indices[:test_size])

    X_train, X_test, y_train, y_test = [], [], [], []
    for i, (row, target) in enumerate(zip(X, y)):
        if i in test_idx:
            X_test.append(row)
            y_test.append(target)
        else:
            X_train.append(row)
            y_train.append(target)
    return X_train, X_test, y_train, y_test


def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("mean requires at least one value")
    return sum(values) / len(values)


def simple_linear_regression(x: list[float], y: list[float]) -> tuple[float, float]:
    """Fit y = intercept + slope*x using plain Python loops."""
    x_bar = mean(x)
    y_bar = mean(y)
    numerator = sum((xi - x_bar) * (yi - y_bar) for xi, yi in zip(x, y))
    denominator = sum((xi - x_bar) ** 2 for xi in x)
    if denominator == 0:
        raise ValueError("cannot fit a line when x has no variation")
    slope = numerator / denominator
    intercept = y_bar - slope * x_bar
    return intercept, slope


def predict_simple(x: list[float], intercept: float, slope: float) -> list[float]:
    return [intercept + slope * xi for xi in x]


def add_intercept(X: list[list[float]]) -> list[list[float]]:
    return [[1.0] + row[:] for row in X]


def standardize_fit(X: list[list[float]]) -> tuple[list[float], list[float]]:
    cols = list(zip(*X))
    means = [mean(list(col)) for col in cols]
    stds = []
    for col, m in zip(cols, means):
        variance = sum((v - m) ** 2 for v in col) / max(1, len(col) - 1)
        std = math.sqrt(variance)
        stds.append(std if std > 0 else 1.0)
    return means, stds


def standardize_transform(
    X: list[list[float]],
    means: list[float],
    stds: list[float],
) -> list[list[float]]:
    return [[(v - m) / s for v, m, s in zip(row, means, stds)] for row in X]


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def predict_matrix(X: list[list[float]], weights: list[float]) -> list[float]:
    return [dot(row, weights) for row in X]


def fit_linear_gradient_descent(
    X: list[list[float]],
    y: list[float],
    learning_rate: float = 0.04,
    epochs: int = 2500,
    l2: float = 0.0,
) -> tuple[list[float], list[float]]:
    """Fit multiple linear regression with gradient descent.

    This is written for teaching: readable loops, clear metric history, and
    optional L2 shrinkage for ridge-style regularization.
    """
    X_design = add_intercept(X)
    n = len(X_design)
    k = len(X_design[0])
    weights = [0.0] * k
    loss_history: list[float] = []

    for _ in range(epochs):
        predictions = predict_matrix(X_design, weights)
        errors = [pred - actual for pred, actual in zip(predictions, y)]
        loss = mean([e * e for e in errors])
        loss_history.append(loss)

        gradients = []
        for j in range(k):
            grad = 2.0 * mean([errors[i] * X_design[i][j] for i in range(n)])
            if j > 0:
                grad += 2.0 * l2 * weights[j]
            gradients.append(grad)

        weights = [w - learning_rate * g for w, g in zip(weights, gradients)]
    return weights, loss_history


def polynomial_features_one_column(x: list[float], degree: int = 2) -> list[list[float]]:
    """Turn one input column into [x, x^2, ..., x^degree]."""
    return [[value ** power for power in range(1, degree + 1)] for value in x]


def mae(y_true: list[float], y_pred: list[float]) -> float:
    return mean([abs(a - p) for a, p in zip(y_true, y_pred)])


def rmse(y_true: list[float], y_pred: list[float]) -> float:
    return math.sqrt(mean([(a - p) ** 2 for a, p in zip(y_true, y_pred)]))


def r2_score(y_true: list[float], y_pred: list[float]) -> float:
    y_bar = mean(y_true)
    ss_res = sum((a - p) ** 2 for a, p in zip(y_true, y_pred))
    ss_tot = sum((a - y_bar) ** 2 for a in y_true)
    return 1.0 - ss_res / ss_tot if ss_tot else 0.0


def print_metrics(name: str, y_true: list[float], y_pred: list[float]) -> None:
    print(f"\n{name}")
    print("-" * len(name))
    print(f"MAE : {mae(y_true, y_pred):.2f}")
    print(f"RMSE: {rmse(y_true, y_pred):.2f}")
    print(f"R2  : {r2_score(y_true, y_pred):.3f}")


def main() -> None:
    X, y = make_house_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # Baseline: predict the training mean for every test example.
    baseline = [mean(y_train)] * len(y_test)
    print_metrics("Baseline mean predictor", y_test, baseline)

    # Simple regression with one feature: house size.
    size_train = [row[0] for row in X_train]
    size_test = [row[0] for row in X_test]
    intercept, slope = simple_linear_regression(size_train, y_train)
    simple_pred = predict_simple(size_test, intercept, slope)
    print_metrics("Simple regression from scratch", y_test, simple_pred)
    print(f"intercept={intercept:.2f}, slope={slope:.4f}")

    # Multiple regression with gradient descent.
    means, stds = standardize_fit(X_train)
    X_train_scaled = standardize_transform(X_train, means, stds)
    X_test_scaled = standardize_transform(X_test, means, stds)
    weights, losses = fit_linear_gradient_descent(X_train_scaled, y_train)
    multi_pred = predict_matrix(add_intercept(X_test_scaled), weights)
    print_metrics("Multiple regression with gradient descent", y_test, multi_pred)
    print("weights:", [round(w, 3) for w in weights])
    print("first loss:", round(losses[0], 2), "last loss:", round(losses[-1], 2))

    # Ridge-style shrinkage using the same gradient descent trainer.
    ridge_weights, _ = fit_linear_gradient_descent(
        X_train_scaled,
        y_train,
        learning_rate=0.04,
        epochs=2500,
        l2=0.05,
    )
    ridge_pred = predict_matrix(add_intercept(X_test_scaled), ridge_weights)
    print_metrics("Ridge-style regression from scratch", y_test, ridge_pred)

    # Polynomial feature example using only size.
    size_mean, size_std = mean(size_train), math.sqrt(
        sum((v - mean(size_train)) ** 2 for v in size_train) / (len(size_train) - 1)
    )
    size_train_scaled = [(v - size_mean) / size_std for v in size_train]
    size_test_scaled = [(v - size_mean) / size_std for v in size_test]
    X_poly_train = polynomial_features_one_column(size_train_scaled, degree=2)
    X_poly_test = polynomial_features_one_column(size_test_scaled, degree=2)
    poly_weights, _ = fit_linear_gradient_descent(X_poly_train, y_train)
    poly_pred = predict_matrix(add_intercept(X_poly_test), poly_weights)
    print_metrics("Polynomial regression from scratch", y_test, poly_pred)


if __name__ == "__main__":
    main()
