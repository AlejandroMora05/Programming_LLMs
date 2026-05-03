import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def preprocesar_muestras_celulares(df, target_col):
    """
    Preprocesa un DataFrame de muestras celulares para ML:
    1) Separa la variable objetivo (y) indicada por target_col.
    2) Construye X usando solo columnas numéricas, excluyendo target_col.
    3) Imputa NaNs con el promedio de cada columna (SimpleImputer).
    4) Estandariza a media 0 y desvío 1 (StandardScaler).
    5) Devuelve (X_procesada, y) como arrays de numpy.
    """
    # 1) Separar y
    y = df[target_col].to_numpy()

    # 2) X solo con columnas numéricas excluyendo la target
    X = df.drop(columns=[target_col])
    X = X.select_dtypes(include=[np.number])

    # 3) Imputación (media)
    imputer = SimpleImputer(strategy="mean")
    X_imputada = imputer.fit_transform(X)

    # 4) Escalado (standardization)
    scaler = StandardScaler()
    X_escalada = scaler.fit_transform(X_imputada)

    # 5) Retornar arrays numpy
    return X_escalada, y