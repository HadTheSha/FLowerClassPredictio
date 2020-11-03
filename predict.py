#import 
import argparse
import matplotlib.pyplot as plt
import numpy as np 
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import json

#Load
data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'


#args: 
parser = argparse.ArgumentParser(description='Enter Training Data ')
parser.add_argument('data_path', type=str, help='data directoroy')
parser.add_argument('--save_directory', type=str, help='Input path for an directory to be saved')
parser.add_argument('--arch', default='SVG16',type=str, help='arch')
parser.add_argument('--learning_Rate', default=0.028,type=str, help='learning rate')
parser.add_argument('--hidden_units',type=str, help='hidden units')
parser.add_argument('--epochs', default=1,type=str, help='epochs')
parser.add_argument('--gpu', default='cpu' ,type=str, help='gpu')
args = parser.parse_args()
data_path=args.data_path
arch=args.arch
CP_dir= args.save_directory
epchs=args.epochs
hiddenUnits=args.hidden_units
lrate=args.learning_Rate
gpu=args.gpu




#Transforms for dirs: 

train_transforms = transforms.Compose([transforms.Resize(255),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 
                                                           0.456,
                                                           0.406],std=[0.229,
                                                                   0.224, 
                                                                   0.225])])

test_transforms = transforms.Compose([transforms.Resize(240),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 
                                                           0.456,
                                                           0.406],std=[0.229,
                                                                   0.224, 
                                                                   0.225])])

valid_transforms = transforms.Compose([transforms.Resize(259),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 
                                                           0.456,
                                                           0.406],std=[0.229,
                                                                   0.224, 
                                                                   0.225])])

image_datasets_train =datasets.ImageFolder(data_path, transform=train_transforms)
image_datasets_test = datasets.ImageFolder(test_dir, transform=test_transforms)
image_datasets_valid = datasets.ImageFolder(valid_dir, transform=valid_transforms)


#The dataloaders
dataloaders_train = torch.utils.data.DataLoader(image_datasets_train, batch_size=64, shuffle=True)
dataloaders_test = torch.utils.data.DataLoader(image_datasets_test, batch_size=64)
dataloaders_valid = torch.utils.data.DataLoader(image_datasets_valid, batch_size=64)


#Load Flowers Classes 
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
#Specify device 
#device = torch.device("cuda" if torch.cuda.is_available() && gpu != 'cpu'  else "cpu")
if(torch.cuda.is_available() and gpu != "cpu"):
    device="cuda"
else:
    device="cpu"


model= models.arch
print(model.parameters)

for param in model.parameters():
    param.requires_grad=False 
     
classifier= nn.Sequential(
    nn.Linear(512 * 7 * 7, 4096),
    nn.ReLU(),
    nn.Dropout(0.9),
    nn.Linear(4096, 4096),
    nn.ReLU(),
    nn.Dropout(0.6),
    nn.Linear(4096, 102),
    nn.LogSoftmax(dim=1)
)
#attaching untrained to my model
model.classifier= classifier
    #Training the model 


criterion =nn.NLLLoss()
optimizer=optim.Adam(model.classifier.parameters(), lr=lrate)

model.to(device)

epochs=epchs
steps=0

running_loss=0
print_every=11

train_losses, valid_losses = [], []

for e in range(epochs):
    for  images , labels in  dataloaders_train:
        steps += 1

        #move them to GPU 
        images, labels= images.to(device), labels.to(device)

        optimizer.zero_grad()

        log_ps=model.forward(images)
        loss=criterion(log_ps,labels)
        loss.backward()
        optimizer.step() 

        #keep track of training loss
        running_loss += loss.item() 

        if steps % print_every ==0:
            model.eval() #turn off dropout
            valid_loss=0 
            accuracy=0 

            with torch.no_grad():
                for images, labels in dataloaders_valid:
                    images, labels = images.to(device), labels.to(device)
                    log_ps=model.forward(images)
                    loss=criterion(log_ps,labels)
                    valid_loss += loss.item() 

                    #accuracy calculations: 
                    ps=torch.exp(log_ps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor))

            train_losses.append(running_loss/len(dataloaders_train))
            valid_losses.append(valid_loss/len(dataloaders_valid))
            print(f"Epoch {e+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Valid loss: {valid_loss/len(dataloaders_valid):.3f}.. "
                  f"Valid accuracy: {accuracy/len(dataloaders_valid):.3f}")

            print(f'Accuracy: {accuracy.item()*100}%')
            running_loss = 0
            model.train()
model.class_to_idx = image_datasets_train.class_to_idx
checkpoint ={
    'input_size': 25088,
    'mid_size': 4096,
    'output_size': 102,
    'batch_size': 64,
    'epoch': epochs,
    'optimizer':optimizer.state_dict,
    'state_dict': model.state_dict(),
    'loss': nn.NLLLoss(),
    'classifier':classifier,
    'class_to_idx': image_datasets_train.class_to_idx,
    'model': models.vgg16(pretrained=True)
}



torch.save(checkpoint, CP_dir/'checkpoint.pth')


