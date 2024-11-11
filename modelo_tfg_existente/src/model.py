import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os

# Configuración del dispositivo (mps para GPU en Mac, cpu si no está disponible)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Configuración del dataset
input_size = (3, 236, 295)  # Formato de PyTorch: (C, H, W)
num_classes = 5
batch_size = 32

# Transformaciones para normalizar y preparar las imágenes
transform = transforms.Compose([
    transforms.Resize((input_size[1], input_size[2])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Definición de un dataset personalizado
class MusicGenreDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.genres = ['pop', 'classical', 'rock', 'metal', 'reggae']
        
        for label, genre in enumerate(self.genres):
            genre_dir = os.path.join(data_dir, genre)
            for file_name in os.listdir(genre_dir):
                if file_name.endswith('.png'):
                    self.image_paths.append(os.path.join(genre_dir, file_name))
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Carga de datos
train_dataset = MusicGenreDataset(data_dir='./output_dir/train', transform=transform)
val_dataset = MusicGenreDataset(data_dir='./output_dir/val', transform=transform)
test_dataset = MusicGenreDataset(data_dir='./output_dir/test', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Definición del modelo
class MusicGenreClassifier(nn.Module):
    def __init__(self, num_classes):
        super(MusicGenreClassifier, self).__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=2)
        )
        
        self.batch_norm = nn.BatchNorm2d(32)
        self.dropout1 = nn.Dropout(0.6)
        
        self.fc1 = nn.Linear(32 * 13 * 17, 1000)  # Ajustar según el tamaño de salida de las capas
        self.dropout2 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, num_classes)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.batch_norm(x)
        x = self.dropout1(x)
        x = x.view(x.size(0), -1)  # Aplanar el tensor
        x = self.fc1(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

# Inicialización del modelo y moverlo al dispositivo
model = MusicGenreClassifier(num_classes=num_classes).to(device)

# Definición del criterio de pérdida y optimizador
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Entrenamiento del modelo
num_epochs = 30
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)  # Mover datos al dispositivo
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}')

    # Validación
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # Mover datos al dispositivo

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_accuracy = correct / total
    print(f'Validation Accuracy: {val_accuracy * 100:.2f}%')

print('Entrenamiento completo.')

# Evaluación en el conjunto de test
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)  # Mover datos al dispositivo

        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
test_accuracy = correct / total
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

# Guardar el modelo
torch.save(model.state_dict(), "model.pt")
print("Modelo guardado como 'model.pt'")
