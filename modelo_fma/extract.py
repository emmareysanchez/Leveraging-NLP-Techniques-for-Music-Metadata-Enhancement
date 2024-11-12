import pandas as pd
import ast
from collections import Counter
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import shutil


# Ruta al archivo raw_tracks.csv
metadata_path = "./fma_medium/fma_metadata/raw_tracks.csv"
data_dir = "./fma_medium/fma_medium"  # Directorio donde están los archivos .mp3 organizados en subcarpetas numeradas
output_dir = "./output_dir"
os.makedirs(output_dir, exist_ok=True)

# Diccionario de mapeo de géneros según las categorías elegidas
genre_mapping = {
    # ROCK
    "Rock": "Rock", "Indie-Rock": "Rock", "Psych-Rock": "Rock", "Noise-Rock": "Rock", 
    "Post-Rock": "Rock", "Krautrock": "Rock", "Space-Rock": "Rock",
    
    # POP
    "Pop": "Pop", "Experimental Pop": "Pop", "Power-Pop": "Pop",
    
    # METAL
    "Metal": "Metal", "Death-Metal": "Metal", "Black-Metal": "Metal",
    
    # FOLK
    "Folk/Acoustic": "Folk", "Psych-Folk": "Folk", "Freak-Folk": "Folk", 
    "Free-Folk": "Folk", "British Folk": "Folk",
    
    # JAZZ
    "Jazz": "Jazz", "Free-Jazz": "Jazz", "Nu-Jazz": "Jazz", "Modern Jazz": "Jazz", 
    "Jazz: Out": "Jazz", "Jazz: Vocal": "Jazz",
    
    # PUNK
    "Punk": "Punk", "Post-Punk": "Punk", "Electro-Punk": "Punk",
    
    # HIP-HOP/RAP
    "Hip-Hop/Rap": "Hip-Hop/Rap", "Hip-Hop Beats": "Hip-Hop/Rap", 
    "Alternative Hip-Hop": "Hip-Hop/Rap", "Abstract Hip-Hop": "Hip-Hop/Rap",
    
    # ELECTRONIC
    "Electronic": "Electronic", "Electroacoustic": "Electronic", "Lo-Fi": "Electronic", 
    "Ambient Electronic": "Electronic", "IDM": "Electronic", "Glitch": "Electronic",
    "Downtempo": "Electronic", "Minimal Electronic": "Electronic", "Breakbeat": "Electronic",
    "Breakcore - Hard": "Electronic", "Drum & Bass": "Electronic", "Bigbeat": "Electronic",
    
    # R&B / SOUL
    "R&B/Soul": "R&B/Soul", "Soul-RnB": "R&B/Soul",
    
    # CLASSICAL MUSIC
    "Classical": "Classical", "20th Century Classical": "Classical", "Chamber Music": "Classical", 
    "Opera": "Classical", "Choral Music": "Classical", "Composed Music": "Classical", "Minimalism": "Classical",
    
    # COUNTRY / AMERICANA
    "Country": "Country/Americana", "Americana": "Country/Americana", 
    "Country & Western": "Country/Americana", "Western Swing": "Country/Americana",
    
    # WORLD MUSIC
    "International": "World", "World/International": "World", "Latin America": "World", 
    "Brazilian": "World", "African": "World", "Balkan": "World", "Middle East": "World", 
    "Indian": "World", "N. Indian Traditional": "World", "South Indian Traditional": "World", 
    "Turkish": "World", "Romany (Gypsy)": "World", "Klezmer": "World", "North African": "World", 
    "Pacific": "World", "Asia-Far East": "World", "Spanish": "World", "Flamenco": "World", 
    "Fado": "World", "Tango": "World", "Cumbia": "World", "Salsa": "World"
}

# Función para mapear el género a su categoría agrupada
def map_genre(genre):
    return genre_mapping.get(genre)

# Leer el archivo de metadatos y filtrar las pistas de los géneros seleccionados
tracks = pd.read_csv(metadata_path)
selected_tracks = {}

for _, row in tracks.iterrows():
    try:
        genres = ast.literal_eval(row['track_genres'])
        for genre in genres:
            genre_name = map_genre(genre['genre_title'])
            if genre_name:  # Solo incluir géneros seleccionados
                track_id = row['track_id']
                selected_tracks[track_id] = genre_name
                break  # Solo el primer género que coincide
    except (ValueError, KeyError):
        continue

# Contador para el progreso
total_tracks = len(selected_tracks)
processed_count = 0


# Generar espectrogramas de Mel para los archivos de los géneros seleccionados
for track_id, genre in selected_tracks.items():
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
        processed_count += 1
        if processed_count % 100 == 0 or processed_count == total_tracks:
            print(f"Processed {processed_count}/{total_tracks} tracks ({(processed_count/total_tracks)*100:.2f}%)")


# Crear directorios de salida para train, val, y test dentro de output_dir
train_dir = os.path.join(output_dir, "train")
val_dir = os.path.join(output_dir, "val")
test_dir = os.path.join(output_dir, "test")

# Crear carpetas de salida para cada conjunto de datos y género
for genre in genre_mapping.values():
    os.makedirs(os.path.join(train_dir, genre), exist_ok=True)
    os.makedirs(os.path.join(val_dir, genre), exist_ok=True)
    os.makedirs(os.path.join(test_dir, genre), exist_ok=True)

# Dividir los archivos en train, val y test para cada género
for genre in genre_mapping.values():
    genre_path = os.path.join(output_dir, genre)
    if os.path.exists(genre_path):
        files = os.listdir(genre_path)
        
        # Dividir en 80% train, 10% val, 10% test
        train_files, temp_files = train_test_split(files, test_size=0.2, random_state=42)
        val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)
        
        # Mover archivos a las carpetas correspondientes
        for file in train_files:
            shutil.move(os.path.join(genre_path, file), os.path.join(train_dir, genre, file))
        for file in val_files:
            shutil.move(os.path.join(genre_path, file), os.path.join(val_dir, genre, file))
        for file in test_files:
            shutil.move(os.path.join(genre_path, file), os.path.join(test_dir, genre, file))

    # Eliminar la carpeta del género original si queda vacía
    if os.path.exists(genre_path) and not os.listdir(genre_path):
        os.rmdir(genre_path)

print("Data preparation completed successfully.")
