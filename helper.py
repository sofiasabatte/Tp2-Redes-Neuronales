import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import io
import mlflow
import mlflow.pytorch
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils

# ---------------------------------------------------------------------------
# Utilidades de logging
# ---------------------------------------------------------------------------

def plot_to_tensorboard(fig, writer, tag, step):
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    image = Image.open(buf).convert("RGB")
    image = np.array(image)
    image = torch.tensor(image).permute(2, 0, 1) / 255.0
    writer.add_image(tag, image, global_step=step)
    plt.close(fig)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ---------------------------------------------------------------------------
# Arquitecturas
# ---------------------------------------------------------------------------

class MLPClassifier(nn.Module):
    """MLP original del proyecto anterior (se conserva para comparación)."""
    def __init__(self, input_size=64*64*3, hidden1=512, hidden2=128, dropout=0.0, num_classes=10):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, num_classes)
        )

    def forward(self, x):
        return self.model(x)


class AlexNetLike(nn.Module):
    """
    CNN inspirada en AlexNet, adaptada a imágenes pequeñas (32x32 o 64x64).

    AlexNet original (Krizhevsky et al., 2012) usa:
      Conv(96, 11x11, stride=4) -> MaxPool -> LRN
      Conv(256, 5x5, pad=2)     -> MaxPool -> LRN
      Conv(384, 3x3, pad=1)
      Conv(384, 3x3, pad=1)
      Conv(256, 3x3, pad=1)     -> MaxPool
      FC(4096) -> FC(4096) -> FC(num_classes)

    Aquí se escalan los filtros y se reemplaza LRN por BatchNorm,
    que es más estable y efectivo en la práctica.
    """
    def __init__(self, input_size=64, dropout=0.5, num_classes=9):
        super().__init__()

        # --- Bloque 1: Conv grande inicial (como AlexNet) ---
        self.features = nn.Sequential(
            # Bloque 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),   # AlexNet usa 11x11 para 224px; adaptamos a 3x3
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                            # -> input_size/2

            # Bloque 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                            # -> input_size/4

            # Bloque 3 (sin pooling, como las 3 capas del medio de AlexNet)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # Bloque 4
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # Bloque 5
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                            # -> input_size/8
        )

        spatial = input_size // 8
        flat_size = 64 * spatial * spatial

        # --- Clasificador denso (análogo a las FC de AlexNet) ---
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class CNNClassifier(nn.Module):
    """CNN simple del proyecto anterior (se conserva para comparación)."""
    def __init__(self, input_size, dropout=0.0, num_classes=10):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1, padding_mode="reflect"),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1, padding_mode="reflect"),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear((input_size // 4) ** 2 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.model(x)
