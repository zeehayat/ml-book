from __future__ import annotations

import json
import shutil
import subprocess
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "ml_regression_coding_video"
SLIDES_DIR = OUT / "slides"
AUDIO_DIR = OUT / "audio"
SEGS_DIR = OUT / "segments"
VIDEO_PATH = OUT / "ML_Regression_for_Beginners_Coding_Tutorial.mp4"
SCRIPT_PATH = OUT / "narration_script.md"
TIMESTAMPS_PATH = OUT / "chapter_timestamps.md"
PLAN_PATH = OUT / "lesson_plan.json"

W, H = 1280, 720
FPS = 24

BG = "#0b1020"
PANEL = "#151d32"
PANEL2 = "#1d2840"
CODE_BG = "#07111f"
WHITE = "#f8fafc"
MUTED = "#b8c3d9"
FAINT = "#7d8aa5"
CYAN = "#22d3ee"
GREEN = "#86efac"
YELLOW = "#facc15"
ORANGE = "#fb923c"
ROSE = "#fb7185"
BLUE = "#93c5fd"
VIOLET = "#c4b5fd"
GRID = "#27364f"

FONT_REG = Path("/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf")
FONT_BOLD = Path("/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf")
FONT_MONO = Path("/usr/share/fonts/truetype/noto/NotoMono-Regular.ttf")
if not FONT_REG.exists():
    FONT_REG = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
if not FONT_BOLD.exists():
    FONT_BOLD = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
if not FONT_MONO.exists():
    FONT_MONO = Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf")


def font(size: int, bold: bool = False, mono: bool = False) -> ImageFont.FreeTypeFont:
    path = FONT_MONO if mono else (FONT_BOLD if bold else FONT_REG)
    return ImageFont.truetype(str(path), size)


SLIDES = [
    {
        "title": "ML Regression for Beginners",
        "kicker": "CODING-FOCUSED VIDEO",
        "kind": "title",
        "duration": 8,
        "subtitle": "Build regression models first in pure Python, then with Python libraries.",
        "narration": "Welcome. In this beginner video, we will learn machine learning regression by coding. We will not focus on mathematical proofs or statistical theory today. We will build models, make predictions, evaluate errors, and then move from pure Python to practical Python libraries.",
    },
    {
        "title": "What Regression Does in Code",
        "kicker": "MENTAL MODEL",
        "kind": "scatter",
        "duration": 9,
        "bullets": [
            "Input: rows of features such as size, bedrooms, and age.",
            "Output: a number, such as house price.",
            "Model: a Python object or function that maps features to predictions.",
            "Training: adjust model settings so predictions become less wrong.",
        ],
        "narration": "Regression predicts a number. The input is a row of features, and the output is a numeric prediction. In code, a model is just a function or object that maps inputs to predictions. Training means changing model settings until the predictions are less wrong.",
    },
    {
        "title": "Video Roadmap",
        "kicker": "WHAT WE WILL BUILD",
        "kind": "pipeline",
        "duration": 8,
        "steps": ["Data", "Split", "Baseline", "Train", "Evaluate", "Improve"],
        "bullets": [
            "Part 1: Pure Python, no external ML packages.",
            "Part 2: NumPy, pandas, scikit-learn, matplotlib, optional statsmodels.",
            "End: advanced topics, use cases, and exercises.",
        ],
        "narration": "The workflow is simple: create data, split it, build a baseline, train a model, evaluate it, and improve it. We will do this twice: first from scratch in pure Python, then with the Python library ecosystem.",
    },
    {
        "title": "Our Example Dataset",
        "kicker": "HOUSE PRICE REGRESSION",
        "kind": "table",
        "duration": 8,
        "headers": ["size_sqft", "bedrooms", "age_years", "region", "price_k"],
        "rows": [
            ["850", "2", "18", "Central", "183"],
            ["1450", "3", "8", "North", "321"],
            ["2100", "4", "12", "South", "401"],
            ["2900", "5", "3", "North", "594"],
        ],
        "bullets": [
            "Features are the columns used for prediction.",
            "Target is the number we want to predict.",
            "Synthetic data keeps the tutorial repeatable.",
        ],
        "narration": "We will use a synthetic house price dataset. Features include size, bedrooms, age, and region. The target is price in thousands. Synthetic data is useful for teaching because everyone gets the same result.",
    },
    {
        "title": "Pure Python: Store the Data",
        "kicker": "NO NUMPY, NO PANDAS",
        "kind": "code",
        "duration": 10,
        "code": """X = [
    [850.0, 2.0, 18.0],
    [1450.0, 3.0, 8.0],
    [2100.0, 4.0, 12.0],
]
y = [183.0, 321.0, 401.0]""",
        "bullets": [
            "Each row is one example.",
            "Each number inside a row is one feature.",
            "The target list stores the answer for each row.",
        ],
        "narration": "In pure Python, we can store features as a list of rows and targets as a separate list. Each row is one example. Each number inside the row is a feature. This is enough to understand how regression works.",
    },
    {
        "title": "Pure Python: Train/Test Split",
        "kicker": "EVALUATION DISCIPLINE",
        "kind": "code",
        "duration": 10,
        "code": """def train_test_split(X, y, test_fraction=0.25, seed=42):
    rng = random.Random(seed)
    indices = list(range(len(X)))
    rng.shuffle(indices)
    test_size = int(len(X) * test_fraction)
    test_idx = set(indices[:test_size])
    # rows in test_idx go to test; the rest go to train""",
        "bullets": [
            "Train set: fit the model.",
            "Test set: estimate future performance.",
            "Never tune choices using the final test answers.",
        ],
        "narration": "Before training, split data into training and test sets. The training set teaches the model. The test set estimates how well it might work on new data. Do not tune your model using the final test answers.",
    },
    {
        "title": "Pure Python: Baseline Model",
        "kicker": "FIRST MODEL TO BEAT",
        "kind": "code",
        "duration": 8,
        "code": """def mean(values):
    return sum(values) / len(values)

baseline_prediction = mean(y_train)
predictions = [baseline_prediction] * len(y_test)""",
        "bullets": [
            "A baseline is a simple reference model.",
            "For regression, predicting the training mean is a useful start.",
            "A real model should beat the baseline.",
        ],
        "narration": "Always start with a baseline. For regression, a simple baseline predicts the average training target for every example. If your advanced model cannot beat this, something is wrong.",
    },
    {
        "title": "Pure Python: Simple Regression",
        "kicker": "ONE FEATURE",
        "kind": "code",
        "duration": 11,
        "code": """def simple_linear_regression(x, y):
    x_bar = mean(x)
    y_bar = mean(y)
    top = sum((xi-x_bar)*(yi-y_bar) for xi, yi in zip(x, y))
    bottom = sum((xi-x_bar)**2 for xi in x)
    slope = top / bottom
    intercept = y_bar - slope * x_bar
    return intercept, slope""",
        "bullets": [
            "Use one feature, such as size_sqft.",
            "Fit an intercept and slope.",
            "Predict with intercept + slope * feature.",
        ],
        "narration": "The simplest regression uses one feature, such as house size. The training function returns an intercept and a slope. Prediction is then just intercept plus slope times the feature.",
    },
    {
        "title": "Pure Python: Metrics",
        "kicker": "HOW WRONG ARE WE?",
        "kind": "code",
        "duration": 10,
        "code": """def mae(y_true, y_pred):
    return mean([abs(a - p) for a, p in zip(y_true, y_pred)])

def rmse(y_true, y_pred):
    return math.sqrt(mean([(a - p)**2
                          for a, p in zip(y_true, y_pred)]))""",
        "bullets": [
            "MAE: average absolute error.",
            "RMSE: punishes large errors more.",
            "Use metrics on test data, not just training data.",
        ],
        "narration": "Metrics tell us how wrong predictions are. Mean absolute error is easy to explain. RMSE punishes larger mistakes more strongly. Always evaluate on data the model did not train on.",
    },
    {
        "title": "Pure Python: Multiple Features",
        "kicker": "GRADIENT DESCENT",
        "kind": "code",
        "duration": 11,
        "code": """def predict_matrix(X, weights):
    return [sum(x*w for x, w in zip(row, weights)) for row in X]

for epoch in range(epochs):
    pred = predict_matrix(X_design, weights)
    errors = [p - actual for p, actual in zip(pred, y)]
    # compute gradients
    # update each weight: weight -= learning_rate * gradient""",
        "bullets": [
            "Multiple features need multiple weights.",
            "Gradient descent improves weights step by step.",
            "Feature scaling makes training much easier.",
        ],
        "narration": "With multiple features, we need multiple weights. Gradient descent improves those weights in many small steps. It works best when features are scaled to similar ranges.",
    },
    {
        "title": "Pure Python: Scaling Features",
        "kicker": "MAKE TRAINING EASIER",
        "kind": "code",
        "duration": 9,
        "code": """means, stds = standardize_fit(X_train)
X_train_scaled = standardize_transform(X_train, means, stds)
X_test_scaled = standardize_transform(X_test, means, stds)""",
        "bullets": [
            "Fit scaling values on training data only.",
            "Apply the same scaling to test data.",
            "Avoid leakage from test data into training.",
        ],
        "narration": "Scaling turns features into comparable units. Fit the scaling values on the training set only. Then apply those same values to the test set. This prevents data leakage.",
    },
    {
        "title": "Pure Python: Improve the Model",
        "kicker": "FEATURE ENGINEERING",
        "kind": "code",
        "duration": 10,
        "code": """def polynomial_features_one_column(x, degree=2):
    return [[value ** power
             for power in range(1, degree + 1)]
            for value in x]

# Example row: [size] -> [size, size**2]""",
        "bullets": [
            "Feature engineering creates better input columns.",
            "Polynomial features let simple models fit curves.",
            "Regularization helps control overfitting.",
        ],
        "narration": "Feature engineering means creating useful input columns. Polynomial features let a simple model fit curved patterns. If you add many features, regularization helps prevent overfitting.",
    },
    {
        "title": "Pure Python: Ridge-Style Regularization",
        "kicker": "ADVANCED BUT PRACTICAL",
        "kind": "code",
        "duration": 10,
        "code": """if j > 0:
    gradient += 2.0 * l2 * weights[j]

weights[j] = weights[j] - learning_rate * gradient""",
        "bullets": [
            "Ridge discourages very large weights.",
            "It often improves stability with many features.",
            "Do not regularize the intercept in beginner examples.",
        ],
        "narration": "Ridge-style regularization adds a small extra push that discourages huge weights. This often improves stability when features are noisy or correlated. In beginner examples, leave the intercept unregularized.",
    },
    {
        "title": "Move to Python Libraries",
        "kicker": "REAL WORKFLOW",
        "kind": "cards",
        "duration": 8,
        "cards": [
            ("NumPy", "fast arrays and linear algebra"),
            ("pandas", "tables, cleaning, categories"),
            ("scikit-learn", "pipelines, validation, ML models"),
            ("matplotlib", "plots and model checks"),
        ],
        "narration": "Pure Python teaches the mechanics. For real projects, use libraries. NumPy handles arrays, pandas handles tables, scikit-learn handles ML workflows, and matplotlib handles plots.",
    },
    {
        "title": "pandas: Create and Inspect Data",
        "kicker": "LIBRARY TRACK",
        "kind": "code",
        "duration": 9,
        "code": """df = make_house_dataframe()
print(df.head())
print(df.info())
print(df.isna().sum())

X = df[[\"size_sqft\", \"bedrooms\", \"age_years\", \"region\"]]
y = df[\"price_k\"]""",
        "bullets": [
            "Inspect before modeling.",
            "Separate features X from target y.",
            "Check missing values and categories.",
        ],
        "narration": "With pandas, start by inspecting the data. Look at the first rows, column types, missing values, and category values. Then separate features X from target y.",
    },
    {
        "title": "scikit-learn: Train/Test Split",
        "kicker": "LIBRARY TRACK",
        "kind": "code",
        "duration": 8,
        "code": """from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)""",
        "bullets": [
            "Same idea as pure Python.",
            "Library version is tested and convenient.",
            "Set random_state for reproducibility.",
        ],
        "narration": "Scikit-learn gives us a tested train-test split function. It is the same idea as our pure Python version, but shorter and safer. Use random state to make the split reproducible.",
    },
    {
        "title": "Pipelines Prevent Leakage",
        "kicker": "PREPROCESSING",
        "kind": "pipeline",
        "duration": 10,
        "steps": ["Impute", "Scale", "One-Hot", "Model", "Predict"],
        "code": """pipeline = Pipeline([
    (\"preprocess\", preprocess),
    (\"model\", RidgeCV(alphas=np.logspace(-3, 3, 40))),
])""",
        "narration": "A pipeline bundles preprocessing and the model into one object. This prevents leakage because imputation, scaling, and one-hot encoding are fitted only on training data inside each fit.",
    },
    {
        "title": "Categorical Features",
        "kicker": "ONE-HOT ENCODING",
        "kind": "code",
        "duration": 9,
        "code": """categorical_pipeline = Pipeline([
    (\"imputer\", SimpleImputer(strategy=\"most_frequent\")),
    (\"onehot\", OneHotEncoder(handle_unknown=\"ignore\",
                              drop=\"first\")),
])""",
        "bullets": [
            "Models need numbers, not text labels.",
            "One-hot encoding turns categories into columns.",
            "handle_unknown='ignore' protects future predictions.",
        ],
        "narration": "Most machine learning models need numeric inputs. One-hot encoding turns a text category like region into numeric columns. The handle unknown option protects the model when new categories appear later.",
    },
    {
        "title": "Train Several Models",
        "kicker": "MODEL COMPARISON",
        "kind": "code",
        "duration": 11,
        "code": """models = {
    \"LinearRegression\": LinearRegression(),
    \"RidgeCV\": RidgeCV(alphas=np.logspace(-3, 3, 40)),
    \"LassoCV\": LassoCV(max_iter=20000),
    \"RandomForestRegressor\": RandomForestRegressor(),
}""",
        "bullets": [
            "Linear regression is the baseline ML regressor.",
            "Ridge and Lasso add regularization.",
            "Random forest captures nonlinear patterns.",
        ],
        "narration": "Once the workflow is ready, try several models. Linear regression is a strong baseline. Ridge and Lasso add regularization. Random forests can capture nonlinear patterns without manually creating polynomial features.",
    },
    {
        "title": "Evaluate Library Models",
        "kicker": "TEST METRICS",
        "kind": "code",
        "duration": 9,
        "code": """pred = pipeline.predict(X_test)
print(\"MAE\", mean_absolute_error(y_test, pred))
print(\"RMSE\", mean_squared_error(y_test, pred) ** 0.5)
print(\"R2\", r2_score(y_test, pred))""",
        "bullets": [
            "Use the same metrics as pure Python.",
            "Compare models on the same test split.",
            "Prefer clear metrics over impressive-looking output.",
        ],
        "narration": "Evaluation is the same idea as pure Python. Predict on the test set, then compute metrics. Compare models on the same split so the comparison is fair.",
    },
    {
        "title": "Cross-Validation",
        "kicker": "BETTER EVALUATION",
        "kind": "pipeline",
        "duration": 10,
        "steps": ["Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5"],
        "code": """results = cross_validate(
    pipeline, X, y, cv=5,
    scoring={\"rmse\": \"neg_root_mean_squared_error\"}
)""",
        "narration": "Cross-validation repeats training and evaluation across several splits. It gives a more stable estimate than a single split, especially when the dataset is not huge.",
    },
    {
        "title": "Advanced Topics to Explore",
        "kicker": "AFTER THE BASICS",
        "kind": "cards",
        "duration": 10,
        "cards": [
            ("Regularization", "Ridge, Lasso, ElasticNet"),
            ("Feature engineering", "polynomials, interactions, domain features"),
            ("Tree models", "RandomForest, GradientBoosting, XGBoost-style tools"),
            ("Deployment", "save model, monitor drift, retrain safely"),
        ],
        "narration": "After the basics, explore regularization, feature engineering, tree-based regressors, and deployment. Real machine learning also includes monitoring model drift and retraining safely.",
    },
    {
        "title": "Regression Use Cases",
        "kicker": "WHERE THIS APPLIES",
        "kind": "cards",
        "duration": 9,
        "cards": [
            ("Real estate", "estimate house prices"),
            ("Retail", "forecast demand or sales"),
            ("Energy", "predict consumption"),
            ("Healthcare", "estimate risk scores or costs"),
            ("Manufacturing", "predict defects or process output"),
            ("Finance", "forecast numeric outcomes carefully"),
        ],
        "narration": "Regression is used anywhere the target is numeric: house prices, demand forecasts, energy consumption, healthcare costs, manufacturing outputs, and finance forecasts.",
    },
    {
        "title": "Exercises",
        "kicker": "PRACTICE TASKS",
        "kind": "bullets",
        "duration": 10,
        "bullets": [
            "Pure Python: add median absolute error as a metric.",
            "Pure Python: add one more feature and compare test RMSE.",
            "Libraries: add a GradientBoostingRegressor and compare models.",
            "Libraries: use cross-validation to tune Ridge alpha.",
            "Project: build a rent-price predictor with your own CSV.",
        ],
        "narration": "Here are exercises. Add a new metric from scratch. Add more features and compare test RMSE. In the library version, try gradient boosting and cross-validated Ridge tuning. Finally, build a small rent-price predictor from your own CSV.",
    },
    {
        "title": "Final Checklist",
        "kicker": "WHAT TO REMEMBER",
        "kind": "bullets",
        "duration": 9,
        "bullets": [
            "Always start with a baseline.",
            "Split data before evaluating model quality.",
            "Fit preprocessing only on training data.",
            "Use clear metrics: MAE, RMSE, and R2.",
            "Move from pure Python understanding to library workflows.",
        ],
        "narration": "Remember the workflow. Start with a baseline. Split data before evaluation. Fit preprocessing only on training data. Use clear metrics. Learn the pure Python mechanics, then use library workflows for real projects.",
    },
]


def ensure_dirs() -> None:
    OUT.mkdir(exist_ok=True)
    for directory in [SLIDES_DIR, AUDIO_DIR, SEGS_DIR]:
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True)


def rounded(draw: ImageDraw.ImageDraw, box, radius=18, fill=PANEL, outline=None, width=2) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def wrap_text(text: str, size: int, max_width: int) -> list[str]:
    avg = max(8, int(size * 0.54))
    lines: list[str] = []
    for paragraph in text.splitlines():
        lines.extend(textwrap.wrap(paragraph, width=max(1, max_width // avg)) or [""])
    return lines


def draw_text_box(
    draw: ImageDraw.ImageDraw,
    text: str,
    xy: tuple[int, int],
    max_width: int,
    max_height: int,
    size: int,
    fill: str = WHITE,
    bold: bool = False,
    mono: bool = False,
    spacing: int = 8,
) -> int:
    x, y = xy
    line_height = size + spacing
    max_lines = max(1, max_height // line_height)
    lines = wrap_text(text, size, max_width)[:max_lines]
    fnt = font(size, bold=bold, mono=mono)
    for line in lines:
        draw.text((x, y), line, font=fnt, fill=fill)
        y += line_height
    return y


def base(slide: dict, idx: int) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(image)
    for x in range(0, W, 64):
        draw.line((x, 0, x, H), fill="#10182b", width=1)
    for y in range(0, H, 64):
        draw.line((0, y, W, y), fill="#10182b", width=1)
    draw.rectangle((0, 0, 14, H), fill=CYAN)
    draw.text((50, 30), slide["kicker"], font=font(17, bold=True), fill=CYAN)
    draw_text_box(draw, slide["title"], (50, 62), 1080, 58, 39, WHITE, bold=True)
    draw.line((50, 124, 1230, 124), fill=GRID, width=2)
    draw.text((50, 682), "ML REGRESSION FOR BEGINNERS  •  CODING TRACK", font=font(14, bold=True), fill=FAINT)
    draw.text((1170, 680), f"{idx:02d}", font=font(18, bold=True), fill=FAINT)
    return image, draw


def draw_bullets(draw: ImageDraw.ImageDraw, bullets: list[str], x=74, y=170, width=1120, size=24) -> None:
    if not bullets:
        return
    available = 500
    step = min(82, max(56, available // len(bullets)))
    for i, item in enumerate(bullets):
        yy = y + i * step
        rounded(draw, (x, yy, x + width, yy + step - 12), 14, PANEL2)
        draw.ellipse((x + 20, yy + 18, x + 40, yy + 38), fill=GREEN)
        draw_text_box(draw, item, (x + 58, yy + 12), width - 80, step - 20, size, WHITE)


def draw_code(draw: ImageDraw.ImageDraw, code: str, x=58, y=164, w=720, h=382, size=21) -> None:
    rounded(draw, (x, y, x + w, y + h), 12, CODE_BG, "#26364f", 2)
    draw.rectangle((x, y, x + w, y + 34), fill="#111827")
    draw.ellipse((x + 16, y + 12, x + 26, y + 22), fill=ROSE)
    draw.ellipse((x + 36, y + 12, x + 46, y + 22), fill=YELLOW)
    draw.ellipse((x + 56, y + 12, x + 66, y + 22), fill=GREEN)
    yy = y + 52
    for line in code.splitlines()[:16]:
        draw.text((x + 22, yy), line, font=font(size, mono=True), fill="#dbeafe")
        yy += size + 7


def draw_scatter(draw: ImageDraw.ImageDraw, x=710, y=180, w=470, h=330) -> None:
    rounded(draw, (x, y, x + w, y + h), 16, "#0d1629", "#26364f", 2)
    ox, oy = x + 55, y + h - 45
    draw.line((ox, oy, x + w - 35, oy), fill=MUTED, width=2)
    draw.line((ox, oy, ox, y + 35), fill=MUTED, width=2)
    points = [(40, 250), (90, 228), (135, 204), (185, 185), (235, 145), (285, 130), (340, 105), (390, 75)]
    draw.line((ox + 20, oy - 20, ox + 390, oy - 195), fill=GREEN, width=4)
    for px, py in points:
        draw.ellipse((ox + px - 7, y + py - 7, ox + px + 7, y + py + 7), fill=CYAN)
    draw.text((x + 108, y + 18), "features → numeric prediction", font=font(19, bold=True), fill=BLUE)
    draw.text((x + 185, y + h - 30), "size_sqft", font=font(14), fill=FAINT)
    draw.text((x + 8, y + 122), "price", font=font(14), fill=FAINT)


def render_slide(slide: dict, idx: int) -> Image.Image:
    image, draw = base(slide, idx)
    kind = slide["kind"]
    if kind == "title":
        draw.text((70, 170), "CODING", font=font(86, bold=True), fill=CYAN)
        draw.text((70, 268), "REGRESSION", font=font(75, bold=True), fill=WHITE)
        draw_text_box(draw, slide["subtitle"], (75, 370), 760, 95, 30, MUTED)
        draw_scatter(draw, 780, 190, 390, 310)
    elif kind == "scatter":
        draw_bullets(draw, slide.get("bullets", []), x=62, y=158, width=585, size=21)
        draw_scatter(draw, 700, 160, 485, 360)
    elif kind == "pipeline":
        steps = slide.get("steps", [])
        y = 210
        if steps:
            total_w = 1080
            gap = 18
            box_w = (total_w - gap * (len(steps) - 1)) // len(steps)
            x0 = 100
            for i, step in enumerate(steps):
                x = x0 + i * (box_w + gap)
                rounded(draw, (x, y, x + box_w, y + 95), 18, PANEL2, CYAN if i % 2 == 0 else GREEN, 2)
                draw_text_box(draw, step, (x + 14, y + 28), box_w - 28, 45, 23, WHITE, bold=True)
                if i < len(steps) - 1:
                    draw.line((x + box_w + 2, y + 48, x + box_w + gap - 2, y + 48), fill=YELLOW, width=4)
        if "code" in slide:
            draw_code(draw, slide["code"], x=220, y=365, w=850, h=175, size=20)
        else:
            draw_bullets(draw, slide.get("bullets", []), x=170, y=360, width=930, size=22)
    elif kind == "table":
        headers = slide["headers"]
        rows = slide["rows"]
        x, y = 64, 170
        col_w = [160, 150, 150, 170, 140]
        row_h = 54
        xx = x
        for header, cw in zip(headers, col_w):
            rounded(draw, (xx, y, xx + cw, y + row_h), 8, "#1f2a44", "#34435f", 1)
            draw_text_box(draw, header, (xx + 10, y + 15), cw - 20, 25, 17, WHITE, bold=True)
            xx += cw
        for r, row in enumerate(rows):
            yy = y + row_h * (r + 1)
            xx = x
            for value, cw in zip(row, col_w):
                rounded(draw, (xx, yy, xx + cw, yy + row_h), 6, PANEL, "#34435f", 1)
                draw_text_box(draw, value, (xx + 10, yy + 15), cw - 20, 25, 17, MUTED)
                xx += cw
        draw_bullets(draw, slide.get("bullets", []), x=840, y=176, width=360, size=19)
    elif kind == "code":
        draw_code(draw, slide["code"])
        draw_bullets(draw, slide.get("bullets", []), x=810, y=170, width=360, size=19)
    elif kind == "cards":
        cards = slide["cards"]
        cols = 2 if len(cards) <= 4 else 3
        card_w = 520 if cols == 2 else 350
        card_h = 135 if len(cards) <= 4 else 118
        x0 = 96 if cols == 2 else 82
        y0 = 180
        for i, (title, desc) in enumerate(cards):
            x = x0 + (i % cols) * (card_w + 36)
            y = y0 + (i // cols) * (card_h + 28)
            rounded(draw, (x, y, x + card_w, y + card_h), 18, PANEL2, CYAN if i % 2 == 0 else GREEN, 2)
            draw_text_box(draw, title, (x + 24, y + 22), card_w - 48, 35, 24, WHITE, bold=True)
            draw_text_box(draw, desc, (x + 24, y + 65), card_w - 48, 45, 18, MUTED)
    elif kind == "bullets":
        draw_bullets(draw, slide["bullets"], x=100, y=170, width=1040, size=24)

    # Burn a concise narration summary at the bottom so the video is useful without audio.
    rounded(draw, (58, 575, 1220, 656), 12, "#111827", "#26364f", 1)
    draw_text_box(draw, slide["narration"], (82, 591), 1110, 52, 18, "#dbeafe")
    return image


def write_metadata() -> None:
    safe_plan = [
        {k: v for k, v in slide.items() if k != "narration"} | {"narration": slide["narration"]}
        for slide in SLIDES
    ]
    PLAN_PATH.write_text(json.dumps(safe_plan, indent=2), encoding="utf-8")
    lines = ["# ML Regression for Beginners - Narration Script\n"]
    for i, slide in enumerate(SLIDES, 1):
        lines.append(f"## Slide {i:02d}: {slide['title']}\n")
        lines.append(slide["narration"] + "\n")
    SCRIPT_PATH.write_text("\n".join(lines), encoding="utf-8")


def write_actual_timestamps(segment_durations: list[float]) -> None:
    seconds = 0.0
    timestamp_lines = ["# Chapter Timestamps\n"]
    for i, (slide, duration) in enumerate(zip(SLIDES, segment_durations), 1):
        total_seconds = int(round(seconds))
        mins, secs = divmod(total_seconds, 60)
        timestamp_lines.append(f"- {mins:02d}:{secs:02d} - Slide {i:02d}: {slide['title']}")
        seconds += duration
    total_seconds = int(round(seconds))
    mins, secs = divmod(total_seconds, 60)
    timestamp_lines.append(f"\nTotal runtime: {mins:02d}:{secs:02d}")
    TIMESTAMPS_PATH.write_text("\n".join(timestamp_lines) + "\n", encoding="utf-8")


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def probe_duration(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    return float(result.stdout.strip())


def build_video() -> None:
    concat_file = SEGS_DIR / "concat.txt"
    concat_lines = []
    segment_durations: list[float] = []
    for i, slide in enumerate(SLIDES, 1):
        image_path = SLIDES_DIR / f"{i:02d}.png"
        narration_text_path = AUDIO_DIR / f"{i:02d}.txt"
        audio_path = AUDIO_DIR / f"{i:02d}.wav"
        segment_path = SEGS_DIR / f"{i:02d}.mp4"
        render_slide(slide, i).save(image_path)
        narration_text_path.write_text(slide["narration"], encoding="utf-8")
        run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                f"flite=textfile={narration_text_path}:voice=slt",
                "-ar",
                "44100",
                "-ac",
                "1",
                str(audio_path),
            ]
        )
        duration = max(float(slide["duration"]), probe_duration(audio_path) + 0.4)
        segment_durations.append(duration)
        run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-loop",
                "1",
                "-i",
                str(image_path),
                "-i",
                str(audio_path),
                "-t",
                f"{duration:.2f}",
                "-vf",
                "scale=1280:720,format=yuv420p",
                "-r",
                str(FPS),
                "-c:v",
                "libx264",
                "-c:a",
                "aac",
                "-b:a",
                "96k",
                "-pix_fmt",
                "yuv420p",
                str(segment_path),
            ]
        )
        concat_lines.append(f"file '{segment_path.resolve()}'")
    concat_file.write_text("\n".join(concat_lines) + "\n", encoding="utf-8")
    write_actual_timestamps(segment_durations)
    run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_file),
            "-c",
            "copy",
            str(VIDEO_PATH),
        ]
    )


def main() -> None:
    ensure_dirs()
    write_metadata()
    build_video()
    print(f"Created {VIDEO_PATH}")
    print(f"Slides: {SLIDES_DIR}")
    print(f"Audio: {AUDIO_DIR}")
    print(f"Narration: {SCRIPT_PATH}")
    print(f"Timestamps: {TIMESTAMPS_PATH}")


if __name__ == "__main__":
    main()
