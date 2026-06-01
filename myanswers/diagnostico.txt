Pregunta 0617
Mi respuesta funciona correctamente en cuanto a la lógica de preprocesamiento.
El error del validador no venía de StandardScaler ni de OneHotEncoder, 
sino de la forma en que se ejecuta la función: el harness llamó a `transformar_datos()`
sin pasar el argumento `df`, por eso apareció `missing 1 required positional argument: 'df'`.

En otras palabras, el problema era de contrato de entrada entre la solución y el validador,
no de la transformación en sí.