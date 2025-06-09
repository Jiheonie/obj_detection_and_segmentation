import urllib
import numpy as np
from PIL import Image
import cv2
import torch
from torchvision import transforms
from argparse import ArgumentParser


def parse_args():
  parser = ArgumentParser(description='Argument of the inference phase')
  parser.add_argument('--image', '-i', type=str, required=True)
  parser.add_argument('--checkpoint', '-c', type=str, default='deeplab_models')

  args = parser.parse_args()
  return args


def putpalette(image, colors):
  final_image = np.zeros((image.shape[0], image.shape[1], 3))
  for r in range(image.shape[0]):
    for c in range(image.shape[1]):
      final_image[r, c] = colors[image[r, c]]
  return final_image.astype(np.uint8)
   

def inference():
  args = parse_args()

  torch.cuda.empty_cache()
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  checkpoint = torch.load('{}/best_model.pt'.format(args.checkpoint))
  model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_mobilenet_v3_large', 
                          weights=checkpoint['weights']).to(device)
  model.eval()

  # url, filename = ("https://github.com/pytorch/hub/raw/master/images/deeplab1.png", "deeplab1.png")
  # try: urllib.URLopener().retrieve(url, filename)
  # except: urllib.request.urlretrieve(url, filename)

  input_image = Image.open(args.image)
  input_image = input_image.convert("RGB")
  preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ])

  input_tensor = preprocess(input_image)
  input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model


  # move the input and model to GPU for speed if available
  if torch.cuda.is_available():
    input_batch = input_batch.to(device)
    model.to(device)

  with torch.no_grad():
    output = model(input_batch)['out'][0]
  output_predictions = output.argmax(0)

  # create a color pallette, selecting a color for each class
  palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
  colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
  colors = (colors % 255).numpy().astype("uint8")

  # plot the semantic segmentation predictions of 21 classes in each color
  r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(input_image.size)
  open_cv_image = np.array(r)

  open_cv_image = putpalette(open_cv_image, colors)

  cv2.imshow('input', np.array(input_image))
  cv2.imshow('image', open_cv_image)
  cv2.waitKey(0)
  cv2.destroyAllWindows()



if __name__ == '__main__':
  inference()