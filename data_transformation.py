import os
from torchvision import transforms
import itertools
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import PIL
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

import program_param as pp

#transform for mydatabase
data_transforms = {
    'Train': transforms.Compose([
        transforms.Resize(pp.input_size),
        #transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation((-15,15)),
        transforms.RandomGrayscale(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'Validation': transforms.Compose([
        transforms.Resize(pp.input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

#transform for camera (live data)
data_transforms_cam = transforms.Compose([
        transforms.Resize(pp.input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
)

image_datasets = {x: ImageFolder(os.path.join(pp.data_dir, x),data_transforms[x])
                  for x in ['Train', 'Validation']}

dataloaders = {x: DataLoader(image_datasets[x], batch_size=pp.batch_size, shuffle=True, num_workers = pp.num_workers)
              for x in ['Train', 'Validation']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['Train', 'Validation']}

ear_class = image_datasets['Train'].classes
num_of_class = len(ear_class)

def preprocess(image):
    """
    Changes in the image (video frame) enabling use in a convolutional network
    :param image:
    :return:
    """
    circut_y = int((640 - pp.size_pict) / 2)
    circut_x = int((480 - pp.size_pict) / 2)
    image = image[circut_x:circut_x + pp.size_pict, circut_y:circut_y + pp.size_pict]
    image_raw = image
    image = PIL.Image.fromarray(image)  # Webcam frames are numpy array format
    # Therefore transform back to PIL image
    image = data_transforms_cam(image)
    image = image.float()
    # image = Variable(image, requires_autograd=True)
    image = image.cuda()
    image = image.unsqueeze(0)  # I don't know for sure but Resnet-50 model seems to only
    # accpets 4-D Vector Tensor so we need to squeeze another
    return image_raw, image


def get_conf_matrix(model, nb_classes, data, normalize = True):
    """
    Plot confusion matrix from the post-training model
    :param model:
    :param nb_classes:
    :param data:
    :param normalize:
    :return:
    """
    confusion_matrix = torch.zeros(nb_classes, nb_classes)
    with torch.no_grad():
        for i, (inputs, classes) in enumerate(data):
            inputs = inputs.to(pp.device)
            classes = classes.to(pp.device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            for t, p in zip(classes.view(-1), preds.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1
    #print(confusion_matrix)

    #classes = image_datasets['Validation'].targets
    classes = list(range(1, nb_classes))
    cm = confusion_matrix
    cmap=plt.cm.Blues

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.show()

def argmax(network_output):
    """
    Extracting predictions from network output data
    :param network_output:
    :return:
    """
    prediction = network_output.cpu()
    prediction = prediction.detach().numpy()
    top_1 = np.argmax(prediction, axis=1)
    score = np.amax(prediction)
    score = '{:6f}'.format(score)
    prediction = top_1[0]
    result = ear_class[prediction]

    return result,score