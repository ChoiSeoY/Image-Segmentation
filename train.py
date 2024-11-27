# train.py
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from class_mapping import load_class_mapping
from dataset import SegmentationDataset
from model import SimpleSegmentationModel

# 경로 설정
csv_path = "path/to/class_dict.csv"
train_image_dir = "path/to/train"
train_label_dir = "path/to/train_label"
val_image_dir = "path/to/validation"
val_label_dir = "path/to/validation_label"
best_model_path = "best_model.pth"

# 클래스 매핑 로드
class_mapping = load_class_mapping(csv_path)
print(f"Loaded class mapping: {class_mapping}")

# Transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 데이터셋 및 데이터로더
train_dataset = SegmentationDataset(train_image_dir, train_label_dir, class_mapping, transform=transform)
val_dataset = SegmentationDataset(val_image_dir, val_label_dir, class_mapping, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# 모델, 손실 함수 및 옵티마이저
num_classes = len(class_mapping)
model = SimpleSegmentationModel(num_classes=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 학습 및 검증 루프
num_epochs = 10
best_val_loss = float('inf')

for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = 0.0
    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    val_loss /= len(val_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

    # Save the best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), best_model_path)
        print(f"Best model saved with Validation Loss: {best_val_loss:.4f}")
