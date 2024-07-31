
# Documentación del Sistema de Recomendación de Películas

link video:
programa funcional Sisitema_ex.py
programa no funcional Sistema_experto.py tiene algunos errores 

## Introducción

Este proyecto implementa un sistema de recomendación de películas utilizando embeddings y un modelo de red neuronal. El sistema permite a los usuarios recibir recomendaciones de películas basadas en sus calificaciones anteriores y también les permite calificar nuevas películas. El modelo se entrena con datos de calificaciones de películas y usuarios, y se actualiza dinámicamente a medida que se añaden nuevas calificaciones.

## Estructura del Código

### Importación de Librerías

```python
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Dot, Flatten, Dense
from tensorflow.keras.models import Model
```

Se importan las librerías necesarias para la manipulación de datos (Pandas, NumPy) y para la construcción del modelo de red neuronal (TensorFlow y Keras).

### Creación de la Base de Datos

```python
def crear_base_datos():
    datos = {
        'userId': [1, 1, 2, 2, 3, 3],
        'movieId': [1, 2, 1, 3, 2, 3],
        'rating': [5, 4, 4, 5, 2, 3]
    }
    return pd.DataFrame(datos)
```

Esta función crea un DataFrame con datos de ejemplo de calificaciones de películas por diferentes usuarios.

### Construcción del Modelo

```python
def construir_modelo(ratings, n_factors=50):
    n_users = ratings['userId'].nunique()
    n_movies = ratings['movieId'].nunique()

    user_input = Input(shape=(1,))
    movie_input = Input(shape=(1,))

    user_embedding = Embedding(input_dim=n_users, output_dim=n_factors, input_length=1)(user_input)
    movie_embedding = Embedding(input_dim=n_movies, output_dim=n_factors, input_length=1)(movie_input)

    user_vector = Flatten()(user_embedding)
    movie_vector = Flatten()(movie_embedding)

    dot_product = Dot(axes=1)([user_vector, movie_vector])

    model = Model(inputs=[user_input, movie_input], outputs=dot_product)
    model.compile(optimizer='adam', loss='mse')

    user_encoder = {user_id: i for i, user_id in enumerate(ratings['userId'].unique())}
    movie_encoder = {movie_id: i for i, movie_id in enumerate(ratings['movieId'].unique())}

    user_indices = ratings['userId'].map(user_encoder)
    movie_indices = ratings['movieId'].map(movie_encoder)

    model.fit([user_indices, movie_indices], ratings['rating'], epochs=10, verbose=1)

    return model, user_encoder, movie_encoder
```

Esta función construye el modelo de recomendación utilizando embeddings para usuarios y películas, y entrena el modelo con las calificaciones disponibles.

### Recomendación de Películas

```python
def recomendar_peliculas(model, user_encoder, movie_encoder, ratings, usuario_id, top_n=5):
    if usuario_id not in user_encoder:
        print("Usuario no encontrado.")
        return []

    user_idx = user_encoder[usuario_id]

    all_movie_ids = np.array(list(movie_encoder.values()))
    user_array = np.array([user_idx] * len(all_movie_ids))
    predictions = model.predict([user_array, all_movie_ids]).flatten()

    movie_indices = predictions.argsort()[-top_n:][::-1]
    movie_ids = [list(movie_encoder.keys())[i] for i in movie_indices]

    return movie_ids
```

Esta función predice las calificaciones para todas las películas no vistas por el usuario y recomienda las mejores.

### Calificación de Películas

```python
def calificar_pelicula(ratings, usuario_id, pelicula_id, calificacion):
    nueva_calificacion = {'userId': usuario_id, 'movieId': pelicula_id, 'rating': calificacion}
    ratings = ratings.append(nueva_calificacion, ignore_index=True)
    return ratings
```

Esta función permite al usuario agregar una nueva calificación a la base de datos.

### Interfaz de Consola

```python
def interfaz_consola():
    ratings = crear_base_datos()
    model, user_encoder, movie_encoder = construir_modelo(ratings)

    while True:
        print("1. Obtener recomendaciones")
        print("2. Calificar una película")
        print("3. Salir")
        opcion = input("Selecciona una opción: ")

        if opcion == '1':
            usuario_id = int(input("Introduce tu ID de usuario: "))
            recomendaciones = recomendar_peliculas(model, user_encoder, movie_encoder, ratings, usuario_id)
            print("Películas recomendadas:", recomendaciones)
        elif opcion == '2':
            usuario_id = int(input("Introduce tu ID de usuario: "))
            pelicula_id = int(input("Introduce el ID de la película: "))
            calificacion = int(input("Introduce tu calificación (1-5): "))
            ratings = calificar_pelicula(ratings, usuario_id, pelicula_id, calificacion)
            model, user_encoder, movie_encoder = construir_modelo(ratings)
            print("Calificación registrada.")
        elif opcion == '3':
            break
        else:
            print("Opción inválida.")
```

Esta función proporciona una interfaz de consola para interactuar con el sistema de recomendación, permitiendo a los usuarios obtener recomendaciones, calificar películas y salir del programa.

## Ejecución del Programa

Para ejecutar el programa, simplemente corre el siguiente bloque de código en un entorno Python:

```python
if __name__ == "__main__":
    interfaz_consola()
```

## Resumen

Este sistema de recomendación de películas es una implementación básica que utiliza embeddings y redes neuronales para predecir calificaciones y proporcionar recomendaciones personalizadas. El sistema también permite actualizar dinámicamente las calificaciones y reentrenar el modelo en consecuencia.
