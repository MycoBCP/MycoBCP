import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import time
import csv
from collections import Counter
from torch import nn
from torch import Tensor
from PIL import Image
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import Compose, Resize, ToTensor
from torchvision.utils import draw_bounding_boxes
from torchvision.io import read_image
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torchsummary import summary
import numpy as np
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.metrics import roc_auc_score
from RandAugment import RandAugment
from os import walk, listdir
from os.path import isfile, join

        
class ConvNet(nn.Module):

    def __init__(self, num_classes=11):
        super(ConvNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(64 * 3 * 3, 512),
            nn.GELU(),
            nn.Dropout(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, num_classes),
            nn.Softmax(dim=1)
        )


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 64 * 3 * 3)
        x = self.classifier[0:4](x)  # Get output from the linear layer before the final classification
        output = x  # Save the raw 256-dimensional values
        x = self.classifier[4:](x)  # Continue to the final classification layer
        return x, output

def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]

def find_classes(dir):
    classes = os.listdir(dir)
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def process_image(img,model,class_labels):
   mean = [0.485, 0.456, 0.406] 
   std = [0.229, 0.224, 0.225]
   transform_norm = transforms.Compose([transforms.Resize((300,300)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean, std)])
   # get normalized image
   img_normalized = transform_norm(img).float()
   img_normalized = img_normalized.unsqueeze_(0)
   img_normalized = img_normalized.to(device)

   with torch.no_grad():
      model.eval()  
      output, raw_features = model(img_normalized)  # Get both outputs
      index = output.data.cpu().numpy().argmax()  # Get predicted class index
      class_name = class_labels[index]
      value = raw_features.data.cpu().numpy()  # Use the raw 256-dimensional features
      return value, class_name


def get_filepaths(test_dir):
    filepaths = [join(test_dir, f) for f in listdir(test_dir) if isfile(join(test_dir, f))]
    return filepaths

def slide_window(img, stepsize, w, h, class_labels, class_idx, num):
    x, y = 0, 0
    x_max, y_max = img.size
    raw_values = []  # This will store the 256-dimensional vectors
    class_names = []
    class_count = []
    count = []
    box = []
    
    while y + h < y_max:
        if x + w + stepsize > x_max:
            x = 0
            y += stepsize
        else:
            sub_img = img.crop((x, y, x + w, y + h))
            raw_features, cn = process_image(sub_img, model, class_labels)
            raw_values.append(raw_features)  # Store the raw 256-dimensional vector for each window
            class_count.append(cn)
            class_names.append(cn)
            count = list(Counter(class_count).items())
            rcount = np.zeros((len(class_labels),), dtype=int)
            for i in range(len(count)):
                rcount[class_idx[count[i][0]]] = count[i][1]
            box.append([x, y, x + w, y + h])
            x += stepsize

    top_idx = sorted(range(len(class_names)), key=lambda i: class_names[i])[-num:]  # Sort by class names if needed
    top_class_names = [class_names[i] for i in top_idx]
    top_boxes = [box[i] for i in top_idx]
    
    return top_class_names, top_boxes, list(rcount), raw_values

def draw_boundary_boxes(img,labels,bbs):
    img = np.asarray(img)
    img = np.transpose(img,(2,0,1))
    img = torch.from_numpy(img)
    boxes = torch.tensor(bbs, dtype=torch.int)
    img = draw_bounding_boxes(img, boxes, labels,
                              width=5, 
                              colors='green', 
                              fill=False, 
                              font='C:\\Windows\\Fonts\\impact', 
                              font_size=40)
    img = torchvision.transforms.ToPILImage()(img)
    return img

num_classes = 11
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = ConvNet(num_classes).to(device)
model = nn.DataParallel(model)
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
checkpoint = torch.load('model_save_name.ckpt')
model.load_state_dict(checkpoint)

class_labels,class_idx=find_classes('BSL2_Training_Data')

test_dir = 'BSL2_Test_Run_XX_Full'
output_dir = 'BSL2_Test_Run_XX_Full_output\\'
filepaths = get_filepaths(test_dir)
rcount_list = []
    
for img_path in filepaths:
    img = Image.open(img_path)
    labels, bbs, rcount, raw_values = slide_window(img, 60, 600, 600, class_labels, class_idx, 5)
    labeled_img = draw_boundary_boxes(img, labels, bbs)
    labeled_img.save(output_dir + os.path.basename(img_path), "tiff")
    
    # For each window, we append a separate row with a unique window number
    for window_num, raw_feature in enumerate(raw_values):
        row_data = [f"{os.path.basename(img_path)}_window{window_num}"]  # Add the filename + window number as a string
        row_data.extend(raw_feature.flatten().tolist())  # Add the flattened 256-dimensional vector for this window
        rcount_list.append(row_data)  # Append row for each window

# Create a header for the CSV
header = ['FILENAME'] + [f'feature_{i}' for i in range(256)]  # Only 256 feature columns since each row is a separate window

# Write to CSV
csv_output_path = 'BSL2_Test_Run_XX_Full_output\\predictions.csv'
with open(csv_output_path, "w", newline='') as f:  # Use newline='' to avoid extra blank lines
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rcount_list) 