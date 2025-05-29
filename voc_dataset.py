import torch
from torchvision.datasets import VOCDetection
from torchvision.transforms import ToTensor, Compose, RandomAffine, Normalize, ColorJitter
from pprint import pprint


class VOCDataset(VOCDetection):
  def __init__(self, root, year='2012', image_set='train', download=False, transform=None):
    super().__init__(root, year, image_set, download)
    self.transform = transform
    self.categories = ["background", "person", "bird", "cat", "cow", "dog", "horse", "sheep", "aeroplane", "bicycle", "boat", "bus", "car", "motorbike", "train", "bottle", "chair", "diningtable", "pottedplant", "sofa", "tvmonitor"]

  def __getitem__(self, index):
    image, data = super().__getitem__(index)

    if self.transform:
      image = self.transform(image)

    objects = data['annotation']['object']
    boxes = []
    labels = []
    for o in objects:
      bbox = o['bndbox']
      x1 = int(bbox['xmin'])
      y1 = int(bbox['ymin'])
      x2 = int(bbox['xmax'])
      y2 = int(bbox['ymax'])
      boxes.append([x1, y1, x2, y2])
      labels.append(self.categories.index(o['name']))

    boxes = torch.FloatTensor(boxes)
    labels = torch.LongTensor(labels)

    return image, {'boxes': boxes, 'labels': labels}


if __name__ == '__main__':
  transform = Compose([
    ToTensor()
  ])

  train_transform = Compose([
    RandomAffine(degrees=(-5, 5), 
                 translate=(0.15, 0.15),
                 scale=(0.85, 1.15),
                 shear=10),
    ColorJitter(brightness=0.125,
                contrast=0.5,
                saturation=0.5,
                hue=0.05),
    # ToTensor(),
    # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ])

  train_dataset = VOCDataset(root='my_pascal_voc', year='2012', image_set='train', download=False, transform=transform)

  image, data = train_dataset.__getitem__(2000)

  # pprint(data['boxes'])
  # pprint(type(data['boxes']))
  # pprint(data['labels'])
  # pprint(type(data['labels']))
  # pprint(image.shape)

  dataset = VOCDetection(root='my_pascal_voc', year='2012', image_set='train', download=False, transform=train_transform)
  image, target = dataset.__getitem__(2000)
  image.show() 