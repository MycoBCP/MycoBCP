import time
import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import os
import matplotlib.pyplot as plt
import math
import numpy as np
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.metrics import roc_auc_score

def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]


# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = 800
num_classes = 11
batch_size = 800
learning_rate = 0.0003

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    
full_dataset = torchvision.datasets.ImageFolder(root='',
                                           transform=transforms.Compose([
                                                   transforms.Resize([300,300]),
                                                   transforms.RandomHorizontalFlip(p=0.5),
                                                   transforms.RandomRotation(180, resample=False, expand=False, center=None),
                                                   transforms.ToTensor(),
                                                   normalize]))

train_dataset,test_dataset = torch.utils.data.random_split(full_dataset, [math.floor(full_dataset*0.7),full_dataset-math.floor(full_dataset*0.7)])


# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True)

def find_classes(dir):
    classes = os.listdir(dir)
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

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
        x = self.classifier(x)
        return x

model = ConvNet().to(device)
model = nn.DataParallel(model)
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)
#checkpoint = torch.load('model_load_name.ckpt')
model.load_state_dict(checkpoint)

# Train the model
total_step = len(train_loader)
t0=time.time()
loss_t=[]

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)
model.zero_grad()
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % 1 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
            loss_t.append(loss.item())
    scheduler.step()
    
# Test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    y_test =[]
    y_score =[]
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0) 
        correct += (predicted == labels).sum().item()
        c = (predicted == labels).cpu().squeeze().numpy()
        y_score.append(outputs.cpu().numpy())
        y_test.append(labels.cpu().numpy())
    print('Test Accuracy of the model on the test images: {} %'.format(100 * correct / total))
    print('seconds:{}'.format(time.time() - t0))

y_s=np.concatenate(y_score,0)
y_t=np.concatenate(y_test,0)

# Save the model checkpoint
torch.save(model.state_dict(), 'model_save_name.ckpt')
plt.plot(loss_t)
auroc=roc_auc_score(y_t, y_s, average='weighted', sample_weight=None, max_fpr=None, multi_class='ovr',labels=None)
print('ROC AUC score: {}'.format(auroc))
