import numpy as np
import os
import torch
from torch import nn, tensor, optim, from_numpy
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms, models
import json
import PIL
from PIL import Image
import argparse
import time

in_features = {"vgg16" : 25088, "densenet121" : 1024}

def transform_data(location = './flowers/'):
    '''
    Arguments : The path of the test dataset
    Return : transformed data   
    '''
    data_dir = location
    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'valid')
    test_dir = os.path.join(data_dir, 'test')

    # transforming train set for more training examples
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])
                                        ])

    # transforming valid set
    valid_transforms = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])
                                        ])

    # transforming test set (same as valid set)
    test_transforms = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])
                                        ])
    
    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    return train_data, valid_data, test_data

def load_data(train_data, valid_data, test_data):
    '''
    Arguments : transformed data  
    Return : load it
    '''

    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)

    return trainloader, validloader, testloader

def check_gpu(gpu):
    '''
    Arguments: in_arg.gpu
    Returns: whether GPU is available or not 
    '''

    if not gpu:
        return torch.device("cpu")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    print("using {}".format(device))
    return device

def set_up_model(arch="vgg16", dropout="0.3", lr=0.001, hidden_units=512, len=102):
    '''
    Arguments: The architecture for the network(resnet50,densenet121,vgg16), the hyperparameters for the network (dropout, learning rate) and use gpu or not
    Returns: Set up model with NLLLoss() and Adam optimizer
    '''

    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif arch == 'densenet121':
        model = models.densenet121(pretrained=True)
    else:
        print("Sorry {} model is not available. Please select vgg16 or densenet121".format(arch))

    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(nn.Linear(in_features[arch], hidden_units),
                              nn.ReLU(),
                              nn.Dropout(dropout),
                              nn.Linear(hidden_units, len),
                              nn.LogSoftmax(dim=1)
                             )
    
    
    model.classifier = classifier
        
    optimizer = optim.Adam(model.classifier.parameters(), lr)
    criterion = nn.NLLLoss()

    return model, criterion, optimizer 

def train_model(model, trainloader, validloader, device, criterion, optimizer, epochs=2, print_every=10):
    '''
    Arguments: model, trainloader, validloader, device, criterion, optimizer, epochs, print_every
    Returns: Nothing
    '''
    model.to(device)
    running_loss = 0

    for epoch in range(epochs):
        steps = 0
        
        for inputs, labels in trainloader:
            start = time.time()
            
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            logps = model(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()    
            
            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                
                for inputs, labels in validloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model(inputs)
                    batch_loss = criterion(logps, labels)

                    valid_loss += batch_loss.item()

                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                
                end = time.time()
                
                print(f"Epoch {epoch+1}/{epochs}.. "
                    f"Train loss: {running_loss/print_every:.3f}.. "
                    f"Validation loss: {valid_loss/len(validloader):.3f}.. "
                    f"Validation accuracy: {accuracy/len(validloader):.3f}.. "
                    f"Time taken: {end - start:.3f}"
                    )
                
                running_loss = 0
                model.train()

def test_model(model, testloader, device, criterion):
    '''
    Arguments: model, testloader, device, criterion=NLLLoss
    Returns: Nothing
    '''

    test_loss = 0
    accuracy = 0
    model.eval()

    start = time.time()

    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)        
        
        logps = model(inputs)
        loss = criterion(logps, labels)

        test_loss += loss.item()

        # Calculate accuracy
        ps = torch.exp(logps)
        _, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item() 
        
    end = time.time()
        
    print(f"Test loss: {test_loss/len(testloader):.3f}.. "
        f"Test accuracy: {accuracy/len(testloader):.3f}.. "
        f"Time taken: {end - start:.3f}"
        )

def save_checkpoint(model, arch, hidden_units, dropout, train_data, len=102, path='checkpoint.pth'):
    '''
    Arguments: The saving path and the hyperparameters of the network
    Returns: Nothing
    '''

    classifier = nn.Sequential(nn.Linear(in_features[arch], hidden_units),
                              nn.ReLU(),
                              nn.Dropout(dropout),
                              nn.Linear(hidden_units, len),
                              nn.LogSoftmax(dim=1)
                             )

    model.class_to_idx = train_data.class_to_idx

    checkpoint = {'arch': model,
                  'classifier' : classifier,
                  'hidden_units':hidden_units,
                  'state_dict': model.state_dict(),
                  'class_to_idx': model.class_to_idx
                 }

    torch.save(checkpoint, path)

def load_checkpoint(path='checkpoint.pth'):
    '''
    Arguments: The path of the checkpoint file
    Returns: The Neural Netowrk with all hyperparameters, weights and biases
    '''

    checkpoint = torch.load(path)
    model = checkpoint['arch']
    model.classifier = checkpoint['classifier']
    model.hidden_units = checkpoint['hidden_units']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
        
    return model

def process_image(image):
    '''
    Arguments: Takes in image path
    Returns: numpy array of RGB image
    '''

    img_pil = Image.open(image)

    # define transforms

    preprocess = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                          std=[0.229, 0.224, 0.225])
                                    ])
    
    img_tensor = preprocess(img_pil)
    np_image = np.array(img_tensor)
    
    return np_image

def predict(image_path, device, model, topk=5):
    '''
    Predict the class (or classes) of an image using a trained model.
    '''
    
    image = from_numpy(process_image(image_path))
    model, image = model.to(device), image.to(device, dtype=torch.float)
    model.eval()
    
    output = model(image.unsqueeze(0)) 
    ps = torch.exp(output)
    
    # getting the topk (=5) probabilites and indexes
    prob = torch.topk(ps, topk)[0].tolist()[0] # probabilities
    index = torch.topk(ps, topk)[1].tolist()[0] # index
    
    idx = []
    for i in range(len(model.class_to_idx.items())):
        idx.append(list(model.class_to_idx.items())[i][0])
        
    classes = []
    for i in range(topk):
        classes.append(idx[index[i]])
    
    return prob, classes

def show_pred(prob, classes, cat_to_name):
    print(prob)
    print([cat_to_name[i] for i in classes])
    print(f"The flower is {cat_to_name[classes[0]]}")