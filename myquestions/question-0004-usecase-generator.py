import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


def ranking_modelos_cv(X, y, cv=5):
    pre = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler())
    ])

    modelos = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForestClassifier": RandomForestClassifier(random_state=42, n_estimators=150),
        "SVC": SVC(probability=True, random_state=42)
    }

    rows = []
    for nombre, mdl in modelos.items():
        pipe = Pipeline([("pre", pre), ("mdl", mdl)])
        scores = cross_validate(
            pipe, X, y, cv=cv,
            scoring={"acc": "accuracy", "f1": "f1", "auc": "roc_auc"},
            n_jobs=-1
        )
        acc_mean = float(np.mean(scores["test_acc"]))
        f1_mean = float(np.mean(scores["test_f1"]))
        auc_mean = float(np.mean(scores["test_auc"]))
        score_global = float(np.mean([acc_mean, f1_mean, auc_mean]))

        rows.append({
            "modelo": nombre,
            "acc_mean": acc_mean,
            "f1_mean": f1_mean,
            "auc_mean": auc_mean,
            "score_global": score_global
        })

    return pd.DataFrame(rows).sort_values("score_global", ascending=False).reset_index(drop=True)


def generar_caso_de_uso_ranking_modelos_cv():
    rng = np.random.default_rng()

    n = int(rng.integers(220, 450))
    p = int(rng.integers(6, 12))
    cols = [f"f{i}" for i in range(p)]

    X = rng.normal(0, 1, size=(n, p))
    lin = 1.2 * X[:, 0] - 0.8 * X[:, 1] + 0.5 * X[:, 2]
    nonlin = 0.7 * (X[:, 3] ** 2) - 0.6 * np.sin(X[:, 4])
    score = lin + nonlin + rng.normal(0, 0.8, size=n)
    y = (score > np.median(score)).astype(int)

    X_df = pd.DataFrame(X, columns=cols)
    X_df[rng.random(X_df.shape) < rng.uniform(0.00, 0.05)] = np.nan

    input = {
        "X": X_df,
        "y": y,
        "cv": int(rng.integers(3, 7))
    }
    output = ranking_modelos_cv(**input)
    return input, output

# Generamos un ejemplo
if __name__ == "__main__":
    input, output = generar_caso_de_uso_ranking_modelos_cv()
    print(input)
    print(output)