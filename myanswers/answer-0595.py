import numpy as np
from sklearn.neighbors import KNeighborsClassifier


def evaluar_knn_pesos(X_train, y_train, X_test, n_vecinos):
	"""Entrena un KNN con weights='distance' y devuelve predicciones para X_test.

	Parámetros:
	- X_train: array-like de forma (n_samples_train, n_features)
	- y_train: array-like de forma (n_samples_train,)
	- X_test: array-like de forma (n_samples_test, n_features)
	- n_vecinos: int, número de vecinos a usar

	Retorna:
	- y_pred: np.ndarray con las etiquetas predichas para X_test
	"""
	clf = KNeighborsClassifier(n_neighbors=n_vecinos, weights='distance')
	clf.fit(X_train, y_train)
	y_pred = clf.predict(X_test)
	return np.asarray(y_pred)


if __name__ == "__main__":
	# Prueba rápida usando el caso de uso del enunciado
	from sklearn.datasets import make_classification

	X, y = make_classification(n_samples=np.random.randint(50, 150),
							   n_features=np.random.randint(4, 10),
							   random_state=np.random.randint(0, 1000))

	split_idx = int(len(X) * 0.8)
	X_train, X_test = X[:split_idx], X[split_idx:]
	y_train = y[:split_idx]
	n_vecinos = np.random.randint(3, 8)

	y_pred = evaluar_knn_pesos(X_train, y_train, X_test, n_vecinos)
	print("n_vecinos:", n_vecinos)
	print("y_pred:", y_pred)

