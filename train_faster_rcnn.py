import torch
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from voc_dataset import VOCDataset
from torchvision.transforms import Compose, ToTensor, Normalize, RandomAffine, ColorJitter
from torch.utils.data import DataLoader
from torch.optim import SGD
from torchmetrics.detection import MeanAveragePrecision
from torch.utils.tensorboard import SummaryWriter
from pprint import pprint
from tqdm import tqdm
import os
import shutil
from argparse import ArgumentParser

def collate_fn(batch):
  images, labels = zip(*batch)
  return list(images), list(labels)


def parse_args():
  parser = ArgumentParser(description='Argument of the training phase')
  parser.add_argument('--path', '-p', type=str, default='my_pascal_voc')
  parser.add_argument('--batch-size', '-b', type=int, default=4)
  parser.add_argument('--year', '-y', type=str, default='2012')
  parser.add_argument('--learning-rate', '-r', type=int, default=0.01)
  parser.add_argument('--momentum', '-m', type=int, default=0.9)
  parser.add_argument('--checkpoint', '-c', type=str, default='trained_models')
  parser.add_argument('--tensorboard', '-t', type=str, default='tensorboard')
  parser.add_argument('--epochs', '-e', type=int, default=100)

  args = parser.parse_args()
  return args


if __name__ == '__main__':
  args = parse_args()

  torch.cuda.empty_cache()
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  train_transform = Compose([
    RandomAffine(degrees=(-5, 5), 
                 translate=(0.15, 0.15),
                 scale=(0.85, 1.15),
                 shear=10),
    ColorJitter(brightness=0.125,
                contrast=0.5,
                saturation=0.5,
                hue=0.05),
    ToTensor(),
    # the faster rcnn already has standardization
    # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ])
  
  test_transform = Compose([
    ToTensor(),
    # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ])

  train_dataset = VOCDataset(
    root=args.path, 
    year=args.year, 
    image_set='train', 
    download=False, 
    transform=train_transform)
    
  train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=1,
    drop_last=True,
    collate_fn=collate_fn)
  
  val_dataset = VOCDataset(
    root=args.path, 
    year=args.year, 
    image_set='val', 
    download=False, 
    transform=test_transform)
    
  val_dataloader = DataLoader(
    dataset=val_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=1,
    drop_last=False,
    collate_fn=collate_fn)

  model = fasterrcnn_mobilenet_v3_large_fpn(weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT, 
                                            trainable_backbone_layers=3)
  in_channels = model.roi_heads.box_predictor.cls_score.in_features
  model.roi_heads.box_predictor = FastRCNNPredictor(in_channels, len(train_dataset.categories))
  model.to(device)

  optimizer = SGD(params=model.parameters(), lr=args.learning_rate, momentum=args.momentum)

  start_epoch = 0
  best_map = -1
  
  if not os.path.exists(args.checkpoint):
    os.mkdir(args.checkpoint)

  if len(os.listdir(args.checkpoint)) != 0:
    checkpoint = torch.load('{}/last_model.pt'.format(args.checkpoint))
    start_epoch = checkpoint['epoch']
    best_map = checkpoint['best_map']
    model.load_state_dict(checkpoint['weights'])
    optimizer.load_state_dict(checkpoint['optimizer'])
  
  if os.path.exists(args.tensorboard):
    shutil.rmtree(args.tensorboard)

  writer = SummaryWriter('tensorboard')

  for epoch in range(start_epoch, args.epochs):
    model.train()
    progress_bar = tqdm(train_dataloader, colour='cyan')
    num_iters = len(train_dataloader)
    for iter, (images, targets) in enumerate(progress_bar):
      images = [image.to(device) for image in images]
      targets = [{'boxes': target['boxes'].to(device), 
                  'labels': target['labels'].to(device)} 
                  for target in targets]

      # Forward
      losses = model(images, targets)

      if isinstance(losses, list):
        print(losses)

      final_losses = sum([loss for loss in losses.values()])    

      # Backward
      optimizer.zero_grad()
      final_losses.backward()
      optimizer.step()

      progress_bar.set_description("Epoch {}/{}: Training: Iterator {}/{}: Loss value: {:.4f}".format(epoch + 1, args.epochs, iter + 1, num_iters, final_losses))
      writer.add_scalar('Training/Loss', final_losses, epoch * num_iters + iter)

    model.eval()
    progress_bar = tqdm(val_dataloader, colour='yellow')
    val_iters = len(val_dataloader)
    metric = MeanAveragePrecision(iou_type="bbox")
    for iter, (images, targets) in enumerate(progress_bar):
      images = [image.to(device) for image in images]
      with torch.no_grad():
        outputs = model(images)
      preds = [{
        'boxes': output['boxes'].to('cpu'),
        'scores': output['scores'].to('cpu'), 
        'labels': output['labels'].to('cpu'),
      } for output in outputs]

      progress_bar.set_description("             Validate: Iterator {}/{}                    ".format(iter + 1, val_iters))
      metric.update(preds, targets)

    result = metric.compute()
    pprint(result)
    writer.add_scalar("Validate/mAP", result['map'], epoch)
    writer.add_scalar("Validate/mAP_50", result['map_50'], epoch)
    writer.add_scalar("Validate/mAP_75", result['map_75'], epoch)

    checkpoint = {
      'epoch': epoch + 1,
      'best_map': best_map,
      'weights': model.state_dict(),
      'optimizer': optimizer.state_dict(),
    }

    torch.save(checkpoint, '{}/last_model.pt'.format(args.checkpoint))
    if result['map'] > best_map:
      checkpoint['best_map'] = result['map']
      torch.save(checkpoint, '{}/best_model.pt'.format(args.checkpoint))

