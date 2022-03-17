
from __future__ import print_function, division

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

plt.ion()

use_gpu = torch.cuda.is_available()
if use_gpu:
    print("Using CUDA")

# Loading this dataset with pytorch is really easy using ImageFolder
# as the labels are specified by the folders names.


# THIS WILL BE YOUR OWN PATH TO THE DATA.
data_dir = r'D:\School\SE\Group_Project\data'
TRAIN = 'train'
VAL = 'val'
TEST = 'test'

# VGG-16 Takes 224x224 images as input, so we resize all of them
data_transforms = {
    TRAIN: transforms.Compose([
        # Data augmentation is a good practice for the train set
        # Here, we randomly crop the image to 224x224 and
        # randomly flip it horizontally (seems like best fit for graphs?)
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ]),
    VAL: transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ]),
    TEST: transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
}

image_datasets = {
    x: datasets.ImageFolder(
        os.path.join(data_dir, x),
        transform=data_transforms[x]
    )
    for x in [TRAIN, VAL, TEST]
}

dataloaders = {
    x: torch.utils.data.DataLoader(
        image_datasets[x], batch_size=8,
        shuffle=True, num_workers=4
    )
    for x in [TRAIN, VAL, TEST]
}

dataset_sizes = {x: len(image_datasets[x]) for x in [TRAIN, VAL, TEST]}

for x in [TRAIN, VAL, TEST]:
    print("Loaded {} images under {}".format(dataset_sizes[x], x))

print("Classes: ")
class_names = image_datasets[TRAIN].classes
print(image_datasets[TRAIN].classes)


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


# Get a batch of training data
# inputs, classes = next(iter(dataloaders[TRAIN]))
# #Make a grid from batch
# out = torchvision.utils.make_grid(inputs)
# imshow(out, title=[class_names[x] for x in classes])

#
# def imshow(inp, title=None):
#     inp = inp.numpy().transpose((1, 2, 0))
#     # plt.figure(figsize=(10, 10))
#     plt.axis('off')
#     plt.imshow(inp)
#     if title is not None:
#         plt.title(title)
#     plt.pause(0.001)
#
#
# #
# def show_databatch(inputs, classes):
#     out = torchvision.utils.make_grid(inputs)
#     imshow(out, title=[class_names[x] for x in classes])
#
#
# # inputs, classes = next(iter(dataloaders[TRAIN]))
# # show_databatch(inputs, classes)
#
#
# def visualize_model(vgg, num_images=6):
#     was_training = vgg.training
#
#     # Set model for evaluation
#     vgg.train(False)
#     vgg.eval()
#
#     images_so_far = 0
#
#     for i, data in enumerate(dataloaders[TEST]):
#         inputs, labels = data
#         size = inputs.size()[0]
#
#         if use_gpu:
#             inputs, labels = Variable(inputs.cuda(), volatile=True), Variable(labels.cuda(), volatile=True)
#         else:
#             inputs, labels = Variable(inputs, volatile=True), Variable(labels, volatile=True)
#
#         outputs = vgg(inputs)
#
#         _, preds = torch.max(outputs.data, 1)
#         predicted_labels = [preds[j] for j in range(inputs.size()[0])]
#
#         print("Ground truth:")
#         show_databatch(inputs.data.cpu(), labels.data.cpu())
#         print("Prediction:")
#         show_databatch(inputs.data.cpu(), predicted_labels)
#
#         del inputs, labels, outputs, preds, predicted_labels
#         torch.cuda.empty_cache()
#
#         images_so_far += size
#         if images_so_far >= num_images:
#             break
#
#     vgg.train(mode=was_training)  # Revert model back to original training state
#
#
# def eval_model(vgg, criterion):
#     since = time.time()
#     avg_loss = 0
#     avg_acc = 0
#     loss_test = 0
#     acc_test = 0
#
#     test_batches = len(dataloaders[TEST])
#     print("Evaluating model")
#     print('-' * 10)
#
#     for i, data in enumerate(dataloaders[TEST]):
#         if i % 100 == 0:
#             print("\rTest batch {}/{}".format(i, test_batches), end='', flush=True)
#
#         vgg.train(False)
#         vgg.eval()
#         inputs, labels = data
#
#         if use_gpu:
#             inputs, labels = Variable(inputs.cuda(), volatile=True), Variable(labels.cuda(), volatile=True)
#         else:
#             inputs, labels = Variable(inputs, volatile=True), Variable(labels, volatile=True)
#
#         outputs = vgg(inputs)
#
#         _, preds = torch.max(outputs.data, 1)
#         loss = criterion(outputs, labels)
#
#         loss_test += loss.data[0]
#         acc_test += torch.sum(preds == labels.data)
#
#         del inputs, labels, outputs, preds
#         torch.cuda.empty_cache()
#
#     avg_loss = loss_test / dataset_sizes[TEST]
#     avg_acc = acc_test / dataset_sizes[TEST]
#
#     elapsed_time = time.time() - since
#     print()
#     print("Evaluation completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
#     print("Avg loss (test): {:.4f}".format(avg_loss))
#     print("Avg acc (test): {:.4f}".format(avg_acc))
#     print('-' * 10)
#
#
# # # Load the pretrained model from pytorch
# vgg16 = models.vgg16_bn()
# # #vgg16.load_state_dict(torch.load("../input/vgg16bn/vgg16_bn.pth"))
# print(vgg16.classifier[6].out_features)  # 1000
#
# # Freeze training for all layers
# for param in vgg16.features.parameters():
#     param.require_grad = False
#
# # Newly created modules have require_grad=True by default
# num_features = vgg16.classifier[6].in_features
# features = list(vgg16.classifier.children())[:-1]  # Remove last layer
# features.extend([nn.Linear(num_features, len(class_names))])  # Add our layer with 4 outputs
# vgg16.classifier = nn.Sequential(*features)  # Replace the model classifier
# print(vgg16)
#
#
#
# # If you want to train the model for more than 2 epochs, set this to True after the first run
# resume_training = False
#
# if resume_training:
#     print("Loading pretrained model..")
#     vgg16.load_state_dict(torch.load('../input/vgg16-transfer-learning-pytorch/VGG16_v2-OCT_Retina.pt'))
#     print("Loaded!")
#
# if use_gpu:
#     vgg16.cuda()  # .cuda() will move everything to the GPU side
#
# criterion = nn.CrossEntropyLoss()
#
# optimizer_ft = optim.SGD(vgg16.parameters(), lr=0.001, momentum=0.9)
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
#
#
#
# print("Test before training")
# eval_model(vgg16, criterion)
