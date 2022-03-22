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
data = datasets.ImageFolder(data_dir, transform=data_transforms)
class_names = data.classes

#check to see if you have a GPU and use it if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # Load the pretrained model from pytorch
vgg16 = models.vgg16_bn()

# #vgg16.load_state_dict(torch.load("../input/vgg16bn/vgg16_bn.pth"))
print(vgg16.classifier[6].out_features)  # 1000

# Freeze training for all layers
for param in vgg16.features.parameters():
    param.require_grad = False

# Newly created modules have require_grad=True by default
    num_features = vgg16.classifier[6].in_features
    features = list(vgg16.classifier.children())[:-1]  # Remove last layer
    features.extend([nn.Linear(num_features, len(class_names))])  # Add our layer with 4 outputs
    vgg16.classifier = nn.Sequential(*features)  # Replace the model classifier

# If you want to train the model for more than 2 epochs,
# set this to True after the first run

    criterion = nn.CrossEntropyLoss()

# optimization (possibly could use atom)
    optimizer_ft = optim.SGD(vgg16.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)



vgg16.load_state_dict(torch.load(r'C:\Users\matth\PycharmProjects\CSCI338_Project\VGG16_graphs.pt'))


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
