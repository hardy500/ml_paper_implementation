from torchvision import transforms
from utils import create_dl
from pathlib import Path

RESOLUTION = (224, 224) # Table 3 from paper
transform = transforms.Compose([
  transforms.Resize(RESOLUTION),
  transforms.ToTensor()
])

def load_data(transform=transform):
  BS = 32
  img_path = Path("data/pizza_steak_sushi")
  train_dir = img_path/"train"
  test_dir = img_path/"test"
  train_dl, test_dl, class_names = create_dl(train_dir, test_dir, transform, BS)
  return train_dl, test_dl, class_names