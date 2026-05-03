import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

def reducir_reportes_texto(df, col_texto, n_componentes):
    """
    Aplica LSA (TF-IDF + TruncatedSVD) a una columna de texto libre.

    Parámetros
    ----------
    df : pandas.DataFrame
        DataFrame con los reportes.
    col_texto : str
        Nombre de la columna que contiene el texto.
    n_componentes : int
        Número de componentes latentes a conservar (dimensión reducida).

    Retorna
    -------
    (matriz_reducida, varianza_explicada) : tuple[np.ndarray, np.ndarray]
        - matriz_reducida: matriz densa (n_filas, n_componentes) con los textos en el espacio LSA.
        - varianza_explicada: array (n_componentes,) con explained_variance_ratio_.
    """
    # 1) Extraer la columna de texto
    X_texto = df[col_texto]

    # 2) Vectorizar con TF-IDF (matriz dispersa)
    vectorizer = TfidfVectorizer(max_features=500)
    matriz_tfidf = vectorizer.fit_transform(X_texto)

    # 3) Reducir dimensión con TruncatedSVD (especial para matrices dispersas)
    svd = TruncatedSVD(n_components=n_componentes, random_state=42)
    matriz_reducida = svd.fit_transform(matriz_tfidf)
    varianza_explicada = svd.explained_variance_ratio_

    # 4) Devolver tupla con numpy arrays
    return np.asarray(matriz_reducida), np.asarray(varianza_explicada)