import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def detectar_drift_ks(df_base, df_nuevo, columnas, alpha=0.05):
    rows = []
    cols_drift = []

    for c in columnas:
        x = df_base[c].dropna().to_numpy()
        y = df_nuevo[c].dropna().to_numpy()

        if len(x) == 0 or len(y) == 0:
            ks_stat, p_value = np.nan, np.nan
            hay_drift = False
        else:
            ks_stat, p_value = ks_2samp(x, y)
            hay_drift = p_value < alpha

        rows.append({
            "columna": c,
            "ks_stat": ks_stat,
            "p_value": p_value,
            "hay_drift": hay_drift
        })
        if hay_drift:
            cols_drift.append(c)

    df_result = pd.DataFrame(rows).sort_values("ks_stat", ascending=False, na_position="last").reset_index(drop=True)
    return df_result, cols_drift


def generar_caso_de_uso_detectar_drift_ks():
    rng = np.random.default_rng()

    n1 = int(rng.integers(80, 180))
    n2 = int(rng.integers(80, 180))
    p = int(rng.integers(3, 7))
    cols = [f"f{i}" for i in range(p)]

    base, nuevo = {}, {}
    drift_cols = set(rng.choice(cols, size=int(rng.integers(1, max(2, p))), replace=False).tolist())

    for c in cols:
        mu = rng.normal(0, 2)
        sigma = rng.uniform(0.5, 2.0)
        base[c] = rng.normal(mu, sigma, n1)

        if c in drift_cols:
            mu2 = mu + rng.uniform(0.8, 2.5)
            sigma2 = sigma * rng.uniform(1.1, 1.8)
        else:
            mu2 = mu + rng.uniform(-0.2, 0.2)
            sigma2 = sigma * rng.uniform(0.9, 1.1)

        nuevo[c] = rng.normal(mu2, sigma2, n2)

    df_base = pd.DataFrame(base)
    df_nuevo = pd.DataFrame(nuevo)

    for df_ in [df_base, df_nuevo]:
        mask = rng.random(df_.shape) < rng.uniform(0.00, 0.04)
        df_[mask] = np.nan

    alpha = float(rng.choice([0.01, 0.03, 0.05, 0.1]))

    input = {
        "df_base": df_base,
        "df_nuevo": df_nuevo,
        "columnas": cols,
        "alpha": alpha
    }
    output = detectar_drift_ks(**input)
    return input, output

if __name__ == "__main__":
    input, output = generar_caso_de_uso_detectar_drift_ks()
    print(input)
    print(output)
    print(output[0])
    print(output[1])