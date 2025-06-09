import os
import shutil
import numpy as np
import cv2
import torch
from torch.optim import SGD
from torchvision.datasets import VOCSegmentation
from torchvision.transforms import ToTensor, Compose, Normalize, Resize
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torchmetrics.classification import MulticlassAccuracy, MulticlassJaccardIndex
from torchmetrics.segmentation import MeanIoU
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm
from argparse import ArgumentParser


class VOCDataset(VOCSegmentation):
  def __init__(self, root, year = "2012", image_set = "train", download = False, transform = None, target_transform = None, transforms = None):
    super().__init__(root, year, image_set, download, transform, target_transform)
    self.classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 
                    'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
  
  def __getitem__(self, index):
    image, label = super().__getitem__(index)
    label = np.array(label, np.int64)
    label[label == 255] = 0
    return image, label


def parse_args():
  parser = ArgumentParser(description='Argument of the training phase')
  parser.add_argument('--path', '-p', type=str, default='my_pascal_voc')
  parser.add_argument('--batch-size', '-b', type=int, default=4)
  parser.add_argument('--year', '-y', type=str, default='2012')
  parser.add_argument('--learning-rate', '-r', type=int, default=0.01)
  parser.add_argument('--momentum', '-m', type=int, default=0.9)
  parser.add_argument('--checkpoint', '-c', type=str, default='deeplab_models')
  parser.add_argument('--tensorboard', '-t', type=str, default='tensorboard')
  parser.add_argument('--epochs', '-e', type=int, default=100)

  args = parser.parse_args()
  return args



def train():
  args = parse_args()

  torch.cuda.empty_cache()
  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


  transform = Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ])
  target_transform = Compose([
    Resize((224, 224)),
  ])

  train_dataset = VOCDataset(root=args.path, 
                             year=args.year, 
                             image_set='train', 
                             download=False, 
                             transform=transform, 
                             target_transform=target_transform)
  train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=args.batch_size, 
    shuffle=True,
    num_workers=6,
  )

  test_dataset = VOCDataset(root=args.path, 
                            year=args.year, 
                            image_set='val', 
                            download=False, 
                            transform=transform, 
                            target_transform=target_transform)
  test_dataloader = DataLoader(
    dataset=test_dataset,
    batch_size=args.batch_size, 
    shuffle=False,
    num_workers=6,
  )

  # model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True).to(device)
  model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_mobilenet_v3_large', 
                         pretrained=True).to(device)
  optimizer = SGD(params=model.parameters(), lr=1e-3, momentum=args.momentum)
  criterion = CrossEntropyLoss()
  acc_metric = MulticlassAccuracy(num_classes=len(train_dataset.classes)).to(device)
  miou_metric = MulticlassJaccardIndex(num_classes=len(train_dataset.classes)).to(device)


  start_e = 0
  best_miou = -1
  best_acc = -1


  if not os.path.exists(args.checkpoint):
    os.mkdir(args.checkpoint)

  if len(os.listdir(args.checkpoint)) != 0:
    checkpoint = torch.load('{}/last_model.pt'.format(args.checkpoint))
    start_e = checkpoint['epoch']
    best_miou = checkpoint['best_miou']
    model.load_state_dict(checkpoint['weights'])
    optimizer.load_state_dict(checkpoint['optimizer'])
  
  if os.path.exists(args.tensorboard):
    shutil.rmtree(args.tensorboard)

  writer = SummaryWriter('tensorboard')

  for e in range(start_e, args.epochs):
    model.train()
    progress_bar = tqdm(train_dataloader, colour='cyan')
    num_iters = len(train_dataloader)
    all_losses = []
    for i, (images, labels) in enumerate(progress_bar):
      images = images.to(device)
      labels = labels.to(device)
      result = model(images)
      output = result['out']
      loss = criterion(output, labels)
      all_losses.append(loss.item())
      avg_loss = np.mean(all_losses)

      progress_bar.set_description("Epoch {}/{}: Training: Iterator {}/{}: Loss value: {:.4f}"
                                   .format(e + 1, args.epochs, i + 1, num_iters, avg_loss))
      writer.add_scalar('Training/Loss', avg_loss, e * num_iters + i)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()


    model.eval()
    progress_bar = tqdm(test_dataloader, colour='red')
    test_acc = []
    test_miou = []
    with torch.no_grad():
      for i, (images, labels) in enumerate(progress_bar):
        images = images.to(device)
        labels = labels.to(device)
        result = model(images)
        output = result['out']
        test_acc.append(acc_metric(output, labels).item())
        test_miou.append(miou_metric(output, labels).item())
    avg_acc = np.mean(test_acc)
    avg_miou = np.mean(test_miou)
    print("Accuracy: {:.4f}. mIOU: {:.4f}".format(avg_acc, avg_miou))
    writer.add_scalar('Test/Accuracy', avg_acc, e)
    writer.add_scalar('Test/mIOU', avg_miou, e)

    checkpoint = {
      'epoch': e + 1,
      'best_miou': best_miou,
      'weights': model.state_dict(),
      'optimizer': optimizer.state_dict(),
    }

    torch.save(checkpoint, '{}/last_model.pt'.format(args.checkpoint))
    if avg_miou > best_miou:
      checkpoint['best_miou'] = avg_miou
      torch.save(checkpoint, '{}/best_model.pt'.format(args.checkpoint))

        

if __name__ == '__main__':
  train()

