from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torch.nn as nn
import copy
import time
from torchvision import datasets, models, transforms

from torch.optim import lr_scheduler
from torch import optim

from torchvision.models import resnet18
from torchvision.models import resnet50
from torchvision.models import resnet152

import program_param as pp
import data_transformation as dt




def train_model(model, criterion, optimizer, scheduler, device = pp.device, num_epochs=25):
    """

    :param model:
    :param criterion:
    :param optimizer:
    :param scheduler:
    :param num_epochs:
    :return:
    """
    since = time.time()
    time_last = since

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['Train', 'Validation']:
            if phase == 'Train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dt.dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'Train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'Train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'Train':
                scheduler.step()

            time_one_epoch = time.time() - time_last
            time_last = time.time()
            epoch_loss = running_loss / dt.dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dt.dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f} Time: {:.0f}m {:.0f}s'.format(
                phase, epoch_loss, epoch_acc,time_one_epoch // 60, time_one_epoch % 60))

            # deep copy the model
            if phase == 'Validation' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


class WorngModelError():
    """Sygnalzie wrong Model choose"""
    def __call__(self, *args, **kwargs):
        print("Wrong model type")

class _StandardCnn(nn.Module):
    """Declaration of own convolutional network"""

    def __init__(self):
        super(_StandardCnn, self).__init__()
        #first ccn layers
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.ReLU())
        #second cnn layers
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2, 0))
        #third cnn layers
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2, 0))

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(128, 512)
        self.drop_out = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, len(dt.ear_class))

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.drop_out(x)
        x = self.fc2(x)
        return x

class CNN():
    """
    Create prefering CNN
    """
    def __init__(self, cnn_type):
        """
        :param cnn_type: choose prefer cnn type:
            "standard" - create standard CNN (4 - layers)
            "resnet18" - create pre-trained CNN (18- layers)
            "resnet50" - create pre-trained CNN (50- layers)
            "resnet152" - create pre-trained CNN (152- lyers)
        """
        self.device = pp.device
        self.model = 0
        self.optimizer = 0
        self.criterion = 0
        self.epoch = 0
        if(cnn_type == "standard"):
            self.model = _StandardCnn()
        elif(cnn_type == "resnet18"):
            self.model = models.resnet18(pretrained=True)
        elif (cnn_type == "resnet50"):
            self.model = models.resnet18(pretrained=True)
        elif (cnn_type == "resnet152"):
            self.model = models.resnet18(pretrained=True)
        elif(cnn_type == "load"):
            self.load()
        else:
            raise WorngModelError

        self.num_ftrs = self.model.fc.in_features

    def __load_checkpoint(self,filepath):
        """

        :param filepath:
        :return:
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model = checkpoint['model']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer = checkpoint['optimizer']
        self.optimizer = self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.criterion = checkpoint['criterion']
        self.epoch = checkpoint['epoch']
        self.model.to(self.device)
        return self.model, self.optimizer, self.criterion, self.epoch

    def load(self):
        """
        Loading of the trained network data
        :param filepath:
        :return:
        """
        self.model, optimizer_ft, criterion, epoch_num = self.__load_checkpoint('ear_model.tar')
        return self.model, optimizer_ft, criterion, epoch_num

    def train_and_save(self,criterion,optimizer_ft,exp_lr_scheduler,epoch_num):
        self.model = train_model(self.model, self.device, criterion, optimizer_ft, exp_lr_scheduler, epoch_num)
        torch.save({
            'model': self.model,
            'epoch': epoch_num,
            'model_state_dict': self.model.state_dict(),
            'optimizer': optimizer_ft,
            'optimizer_state_dict': optimizer_ft.state_dict(),
            'criterion': criterion,
            'device': self.device
        }, 'ear_model.tar')
        return self.model

    def get_model_param(self):
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer_ft = optim.SGD(self.model.get_model().parameters(), lr=0.001, momentum=0.9)
        self.exp_lr_scheduler = lr_scheduler.StepLR(self.optimizer_ft, step_size=7, gamma=0.1)
        return self.criterion, self.optimizer_ft, self.exp_lr_scheduler


    def get_model(self):
        return self.model

    def get_feature_num(self):
        return self.num_ftrs

