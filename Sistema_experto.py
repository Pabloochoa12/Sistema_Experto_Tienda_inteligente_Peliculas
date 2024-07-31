import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Dot, Flatten, Dense
from tensorflow.keras.models import Model
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Función para crear un DataFrame de calificaciones con datos de ejemplo
def crear_datos():
    # Datos de ejemplo
    datos = {
        'userId': [1, 1, 2, 2, 3, 3],
        'movieTitle': ['Matrix', 'Inception', 'Matrix', 'Avatar', 'Inception', 'Avatar'],
        'rating': [5, 4, 4, 5, 2, 3]
    }
    
    return pd.DataFrame(datos)

# Función para construir y entrenar el modelo
def construir_modelo(ratings):
    # Asegurarse de que hay suficientes datos para dividir
    if len(ratings) < 2:
        raise ValueError("Se requieren al menos 2 registros para dividir en conjuntos de entrenamiento y prueba.")

    # Dividir datos en entrenamiento y prueba
    train, test = train_test_split(ratings, test_size=0.2, random_state=42)
    
    # Codificar las entradas categóricas
    user_encoder = LabelEncoder()
    movie_encoder = LabelEncoder()
    
    # Ajustar los codificadores solo en los datos de entrenamiento
    train['userId'] = user_encoder.fit_transform(train['userId'])
    train['movieTitle'] = movie_encoder.fit_transform(train['movieTitle'])
    
    # Aplicar la misma codificación a los datos de prueba
    test['userId'] = user_encoder.transform(test['userId'])
    test['movieTitle'] = movie_encoder.transform(test['movieTitle'])
    
    # Verificar que todos los IDs en el conjunto de prueba estén en el conjunto de entrenamiento
    if not set(test['userId']).issubset(set(train['userId'])):
        raise ValueError("El conjunto de prueba contiene IDs de usuario no vistos en el conjunto de entrenamiento.")
    if not set(test['movieTitle']).issubset(set(train['movieTitle'])):
        raise ValueError("El conjunto de prueba contiene títulos de película no vistos en el conjunto de entrenamiento.")
    
    num_users = len(user_encoder.classes_)
    num_movies = len(movie_encoder.classes_)
    
    user_input = Input(shape=(1,))
    movie_input = Input(shape=(1,))
    
    user_embedding = Embedding(input_dim=num_users, output_dim=10)(user_input)
    movie_embedding = Embedding(input_dim=num_movies, output_dim=10)(movie_input)
    
    dot_product = Dot(axes=2)([user_embedding, movie_embedding])
    flatten = Flatten()(dot_product)
    
    output = Dense(1)(flatten)
    
    model = Model(inputs=[user_input, movie_input], outputs=output)
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    model.fit([train['userId'], train['movieTitle']], train['rating'], epochs=10, verbose=1)
    return model, user_encoder, movie_encoder

# Función para predecir y recomendar películas
def recomendar_pelicula(model, user_encoder, movie_encoder, ratings, usuario_id, top_n=5):
    # Verificar que el ID del usuario esté en los datos de entrenamiento
    if usuario_id not in user_encoder.classes_:
        raise ValueError("ID de usuario no encontrado en los datos de entrenamiento.")
    
    movie_titles = ratings['movieTitle'].unique()
    user_id_encoded = user_encoder.transform([usuario_id])[0]
    movie_ids_encoded = movie_encoder.transform(movie_titles)
    
    user_ids = np.array([user_id_encoded] * len(movie_titles))
    predictions = model.predict([user_ids, movie_ids_encoded])
    predicted_ratings = np.squeeze(predictions)
    recommended_movie_titles = movie_titles[np.argsort(predicted_ratings)[-top_n:]]
    return recommended_movie_titles

# Función principal para interactuar con el usuario
def main():
    print("Bienvenido al sistema de recomendación de películas.")
    
    # Crear datos con datos de ejemplo
    ratings = crear_datos()
    
    if ratings.empty:
        print("No se ingresaron datos. Salida del programa.")
        return
    
    try:
        # Construir y entrenar el modelo
        model, user_encoder, movie_encoder = construir_modelo(ratings)
    except ValueError as e:
        print(f"Error al construir el modelo: {e}")
        return
    
    # Interactuar con el usuario
    while True:
        try:
            usuario_id = int(input("Introduce el ID del usuario para obtener recomendaciones (o -1 para salir): "))
            if usuario_id == -1:
                print("Gracias por usar el sistema. ¡Hasta luego!")
                break
            if usuario_id not in ratings['userId'].unique():
                print("El ID de usuario no se encuentra en los datos. Inténtalo de nuevo.")
                continue
            
            top_n = int(input("¿Cuántas recomendaciones deseas recibir? "))
            
            recomendaciones = recomendar_pelicula(model, user_encoder, movie_encoder, ratings, usuario_id, top_n)
            print(f"Películas recomendadas para el usuario {usuario_id}: {recomendaciones}")
        
        except ValueError:
            print("Por favor, introduce un número válido.")

if __name__ == "__main__":
    main()

