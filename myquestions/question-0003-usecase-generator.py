import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss


def entrenar_modelo_calibrado_texto_numerico(df, text_col, num_cols, target_col, test_size=0.2, random_state=42):
    data = df.dropna(subset=[target_col]).copy()
    X = data[[text_col] + num_cols]
    y = data[target_col].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    pre = ColumnTransformer(
        transformers=[
            ("txt", TfidfVectorizer(max_features=300), text_col),
            ("num", Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("sc", StandardScaler())
            ]), num_cols)
        ]
    )

    base = Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    model = CalibratedClassifierCV(estimator=base, method="sigmoid", cv=3)
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    return {
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "brier": float(brier_score_loss(y_test, proba)),
        "modelo_calibrado": model
    }


def generar_caso_de_uso_entrenar_modelo_calibrado_texto_numerico():
    rng = np.random.default_rng()
    n = int(rng.integers(180, 350))

    positive_words = ["excelente", "bueno", "rapido", "recomendado", "feliz", "genial"]
    negative_words = ["malo", "lento", "defectuoso", "decepcion", "terrible", "pesimo"]
    neutral_words = ["producto", "servicio", "compra", "cliente", "calidad", "precio"]

    texts, y, word_count, rating = [], [], [], []

    for _ in range(n):
        cls = int(rng.integers(0, 2))
        y.append(cls)
        n_words = int(rng.integers(5, 15))
        tokens = []

        for _ in range(n_words):
            r = rng.random()
            if cls == 1:
                tokens.append(rng.choice(positive_words if r < 0.5 else neutral_words if r < 0.8 else negative_words))
            else:
                tokens.append(rng.choice(negative_words if r < 0.5 else neutral_words if r < 0.8 else positive_words))

        texts.append(" ".join(tokens))
        word_count.append(n_words)
        rating.append(np.clip((4.2 if cls == 1 else 2.0) + rng.normal(0, 0.8), 1, 5))

    df = pd.DataFrame({
        "review_text": texts,
        "word_count": word_count,
        "rating": rating,
        "sentiment": y
    })

    df.loc[rng.random(n) < rng.uniform(0.02, 0.08), "rating"] = np.nan
    df.loc[rng.random(n) < rng.uniform(0.0, 0.03), "sentiment"] = np.nan

    input = {
        "df": df,
        "text_col": "review_text",
        "num_cols": ["word_count", "rating"],
        "target_col": "sentiment",
        "test_size": float(rng.choice([0.2, 0.25, 0.3])),
        "random_state": int(rng.integers(0, 10000))
    }
    output = entrenar_modelo_calibrado_texto_numerico(**input)
    return input, output

if __name__ == "__main__":
    input, output = generar_caso_de_uso_entrenar_modelo_calibrado_texto_numerico()
    print(input)
    print(output)
    print(output["modelo_calibrado"])
    print(output["modelo_calibrado"].predict_proba(
        pd.DataFrame({
            "review_text": ["muy bueno producto"],
            "word_count": [5],
            "rating": [4.5]
        })
    ))