import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def transformar_datos(df):
    """
    Detecta columnas numéricas y categóricas en un DataFrame,
    aplica StandardScaler a numéricas y OneHotEncoder a categóricas,
    y devuelve la matriz resultante.
    """
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