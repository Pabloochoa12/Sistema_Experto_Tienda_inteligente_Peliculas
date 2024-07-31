import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Dot, Flatten, Dense
from tensorflow.keras.models import Model

def crear_base_datos():
    # Crear un DataFrame con datos de ejemplo
    datos = {
        'userId': [1, 1, 2, 2, 3, 3],
        'movieId': [1, 2, 1, 3, 2, 3],
        'rating': [5, 4, 4, 5, 2, 3]
    }
    return pd.DataFrame(datos)

def construir_modelo(ratings, n_factors=50):
    # Encontrar el número de usuarios y películas únicas
    n_users = ratings['userId'].nunique()
    n_movies = ratings['movieId'].nunique()

    # Crear capas de entrada
    user_input = Input(shape=(1,))
    movie_input = Input(shape=(1,))

    # Crear capas de embeddings para usuarios y películas
    user_embedding = Embedding(input_dim=n_users, output_dim=n_factors, input_length=1)(user_input)
    movie_embedding = Embedding(input_dim=n_movies, output_dim=n_factors, input_length=1)(movie_input)

    # Aplanar los embeddings
    user_vector = Flatten()(user_embedding)
    movie_vector = Flatten()(movie_embedding)

    # Calcular el producto punto entre los embeddings
    dot_product = Dot(axes=1)([user_vector, movie_vector])

    # Crear el modelo
    model = Model(inputs=[user_input, movie_input], outputs=dot_product)
    model.compile(optimizer='adam', loss='mse')

    # Crear encoders para mapear IDs a índices en los embeddings
    user_encoder = {user_id: i for i, user_id in enumerate(ratings['userId'].unique())}
    movie_encoder = {movie_id: i for i, movie_id in enumerate(ratings['movieId'].unique())}

    # Convertir IDs a índices
    user_indices = ratings['userId'].map(user_encoder)
    movie_indices = ratings['movieId'].map(movie_encoder)

    # Entrenar el modelo
    model.fit([user_indices, movie_indices], ratings['rating'], epochs=10, verbose=1)

    return model, user_encoder, movie_encoder

def recomendar_peliculas(model, user_encoder, movie_encoder, ratings, usuario_id, top_n=5):
    # Verificar si el usuario existe en los datos
    if usuario_id not in user_encoder:
        print("Usuario no encontrado.")
        return []

    # Obtener el índice del usuario
    user_idx = user_encoder[usuario_id]

    # Predecir calificaciones para todas las películas
    all_movie_ids = np.array(list(movie_encoder.values()))
    user_array = np.array([user_idx] * len(all_movie_ids))
    predictions = model.predict([user_array, all_movie_ids]).flatten()

    # Obtener IDs de las películas con las mejores predicciones
    movie_indices = predictions.argsort()[-top_n:][::-1]
    movie_ids = [list(movie_encoder.keys())[i] for i in movie_indices]

    return movie_ids

def calificar_pelicula(ratings, usuario_id, pelicula_id, calificacion):
    # Agregar una nueva calificación a la base de datos
    nueva_calificacion = {'userId': usuario_id, 'movieId': pelicula_id, 'rating': calificacion}
    ratings = ratings.append(nueva_calificacion, ignore_index=True)
    return ratings

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
            # Reentrenar el modelo con los nuevos datos
            model, user_encoder, movie_encoder = construir_modelo(ratings)
            print("Calificación registrada.")
        elif opcion == '3':
            break
        else:
            print("Opción inválida.")

if __name__ == "__main__":
    interfaz_consola()



