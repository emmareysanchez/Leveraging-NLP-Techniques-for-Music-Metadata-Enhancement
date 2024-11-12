import kagglehub
import os
import zipfile
import shutil
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd


# path = kagglehub.dataset_download("imsparsh/fma-free-music-archive-small-medium", files=["fma_medium.zip"])
# print("Path to dataset files:", path)

# Definir rutas de entrada y salida
data_dir = "./fma_medium"  # Directorio donde están los archivos .mp3 organizados en subcarpetas numeradas
metadata_path = "./fma_medium/fma_metadata/tracks.csv"  # Ruta a tracks.csv
output_dir = "./output_dir"
os.makedirs(output_dir, exist_ok=True)

# Paso 1: Cargar el archivo de metadatos para obtener géneros
tracks = pd.read_csv(metadata_path, index_col=0, header=[0, 1])

# Paso 2: Seleccionar los géneros principales y crear un diccionario de género por track_id
# Los géneros están en la columna 'track' > 'genre_top'
genres = [
    'Blues', 'Classical', 'Country', 'Disco', 'Hip-Hop', 'Jazz', 'Metal', 
    'Pop', 'Reggae', 'Rock', 'Electronic', 'Folk', 'International', 
    'Instrumental', 'Experimental', 'Soul-RnB'
]
tracks = tracks[tracks[('track', 'genre_top')].isin(genres)]  # Filtrar solo las canciones de géneros seleccionados
genre_dict = tracks[('track', 'genre_top')].to_dict()  # Diccionario de track_id a género

# Paso 3: Generar espectrogramas de Mel para cada archivo en el género correspondiente
for track_id, genre in genre_dict.items():
    # Formatear el track_id para encontrar el archivo en la estructura de carpetas
    track_id_str = f"{track_id:06d}"  # Convertir a un string con ceros iniciales (ej. 000123)
    folder = track_id_str[:3]  # Carpeta está determinada por los primeros tres dígitos (ej. 000, 001, ...)
    file_path = os.path.join(data_dir, folder, f"{track_id_str}.mp3")
    
    if os.path.exists(file_path):  # Asegurarse de que el archivo exista
        output_genre_dir = os.path.join(output_dir, genre)
        os.makedirs(output_genre_dir, exist_ok=True)
        
        # Cargar el archivo de audio y generar el espectrograma
        y, sr = librosa.load(file_path, sr=22050)
        S = librosa.feature.melspectrogram(y=y, sr=sr)
        S_dB = librosa.power_to_db(S, ref=np.max)
        
        # Guardar el espectrograma como imagen
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Mel Spectrogram - {genre}')
        plt.tight_layout()
        
        # Nombre del archivo de salida
        output_path = os.path.join(output_genre_dir, f"{track_id_str}.png")
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()

        

print("Espectrogramas generados y organizados por género.")