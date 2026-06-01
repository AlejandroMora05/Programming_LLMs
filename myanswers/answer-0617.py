import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def transformar_datos(df=None):
    """
    Detecta columnas numéricas y categóricas en un DataFrame,
    aplica StandardScaler a numéricas y OneHotEncoder a categóricas,
    y devuelve la matriz resultante.
    """
    if df is None:
        entrada = globals().get("input")
        if isinstance(entrada, dict) and "df" in entrada:
            df = entrada["df"]
        else:
            raise TypeError("transformar_datos() missing 1 required positional argument: 'df'")

    X = df.copy()

    # Detectar columnas
    num_cols = X.select_dtypes(include=[np.number]).columns
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns

    # Crear transformador
    transformer = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ],
        remainder="drop",
    )

    return transformer.fit_transform(X)