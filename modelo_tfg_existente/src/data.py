import os
import shutil
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

# Ruta donde están los audios y donde se guardarán los espectrogramas
data_dir = "./data/genres_original"
output_dir = "./output_dir"

# Asegúrate de que la carpeta de salida exista
os.makedirs(output_dir, exist_ok=True)

# Generar espectrogramas de Mel para cada archivo en cada género
genres = ['pop', 'classical', 'rock', 'metal', 'reggae']  # Los géneros seleccionados
for genre in genres:
    genre_dir = os.path.join(data_dir, genre)
    output_genre_dir = os.path.join(output_dir, genre)
    os.makedirs(output_genre_dir, exist_ok=True)
    
    for filename in os.listdir(genre_dir):
        if filename.endswith(".wav"):
            file_path = os.path.join(genre_dir, filename)
            y, sr = librosa.load(file_path, sr=22050)
            
            # Generar espectrograma de Mel
            S = librosa.feature.melspectrogram(y=y, sr=sr)
            S_dB = librosa.power_to_db(S, ref=np.max)
            
            # Guardar el espectrograma como imagen
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
            plt.colorbar(format='%+2.0f dB')
            plt.title(f'Mel Spectrogram - {genre}')
            plt.tight_layout()
            
            # Nombre del archivo de salida
            output_path = os.path.join(output_genre_dir, filename.replace(".wav", ".png"))
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
            plt.close()

# Directorios de salida para train, val, y test dentro de output_dir
train_dir = os.path.join(output_dir, "train")
val_dir = os.path.join(output_dir, "val")
test_dir = os.path.join(output_dir, "test")

# Crear las carpetas de salida para cada conjunto de datos y género
for genre in genres:
    os.makedirs(os.path.join(train_dir, genre), exist_ok=True)
    os.makedirs(os.path.join(val_dir, genre), exist_ok=True)
    os.makedirs(os.path.join(test_dir, genre), exist_ok=True)

# Dividir los archivos en train, val y test
for genre in genres:
    genre_path = os.path.join(output_dir, genre)
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

# Eliminar las carpetas de géneros vacías
for genre in genres:
    genre_path = os.path.join(output_dir, genre)
    if os.path.exists(genre_path) and not os.listdir(genre_path):
        os.rmdir(genre_path)

   