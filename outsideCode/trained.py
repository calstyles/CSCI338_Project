import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy



#This function requires a PIL(pillow library) image, NOT A FILE PATH
def predict_image(image):
    image_tensor = data_transforms(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input = Variable(image_tensor)
    input = input.to(device)
    output = model(input)
    index = output.data.cpu().numpy().argmax()
    return index


def get_random_images(num):
    data = datasets.ImageFolder(data_dir, transform=data_transforms)
    classes = data.classes
    indices = list(range(len(data)))
    np.random.shuffle(indices)
    idx = indices[:num]
    from torch.utils.data.sampler import SubsetRandomSampler
    sampler = SubsetRandomSampler(idx)
    loader = torch.utils.data.DataLoader(data,
                   sampler=sampler, batch_size=num)
    dataiter = iter(loader)
    images, labels = dataiter.next()
    return images, labels

#change this to your path, We don't know the train and VAl I just like
#to keep it consistent
data_dir = r'D:\School\SE\Group_Project\data'
TRAIN = 'train'
VAL = 'val'
TEST = 'test'


#apply the same transforms to the image as the training.
data_transforms = {
    TEST: transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
}

#Why is this needed outside the function?
#data = datasets.ImageFolder(data_dir, transform=data_transforms)
#classes = data.classes

#check to see if you have a GPU and use it if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load('VGG16_graphs.pt')

#set the model to evaluation mode as to not change parameters
model.eval()




to_pil = transforms.ToPILImage()
images, labels = get_random_images(5)
fig=plt.figure(figsize=(10,10))
for ii in range(len(images)):
    image = to_pil(images[ii])
    index = predict_image(image)
    sub = fig.add_subplot(1, len(images), ii+1)
    res = int(labels[ii]) == index
    sub.set_title(str(classes[index]) + ":" + str(res))
    plt.axis('off')
    plt.imshow(image)
plt.show()
