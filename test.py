# test.py
import torch
from dataset import SegmentationDataset
from model import SimpleSegmentationModel
from class_mapping import load_class_mapping
from torch.utils.data import DataLoader
from torchvision import transforms

# 테스트 모델 함수
def test_model(model, data_loader):
    """
    Evaluate the model on the test dataset.

    Args:
        model (nn.Module): Trained segmentation model.
        data_loader (DataLoader): Test dataset DataLoader.
    """
    model.eval()
    iou_scores = []
    with torch.no_grad():
        for images, labels in data_loader:
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)  # Convert probabilities to class IDs
            intersection = (preds & labels).float().sum((1, 2))
            union = (preds | labels).float().sum((1, 2))
            iou = (intersection / union).mean().item()
            iou_scores.append(iou)
    mean_iou = torch.mean(torch.tensor(iou_scores))
    print(f"Mean IoU on Test Data: {mean_iou:.4f}")

if __name__ == "__main__":
    # 경로 설정
    csv_path = "path/to/class_dict.csv"
    test_image_dir = "path/to/test"
    test_label_dir = "path/to/test_label"
    best_model_path = "best_model.pth"

    # 클래스 매핑 로드
    class_mapping = load_class_mapping(csv_path)
    print(f"Loaded class mapping: {class_mapping}")

    # Transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Test Dataset 및 DataLoader
    test_dataset = SegmentationDataset(test_image_dir, test_label_dir, class_mapping, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    # 모델 초기화 및 가중치 로드
    num_classes = len(class_mapping)
    model = SimpleSegmentationModel(num_classes=num_classes)
    model.load_state_dict(torch.load(best_model_path))
    print("Loaded best model for testing.")

    # 테스트 수행
    test_model(model, test_loader)
