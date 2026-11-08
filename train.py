import argparse

import numpy as np
from tqdm import tqdm  #bar de chargement bibliothèque
from utils._utils import make_data_loader
from model import ResNet34CustomInput

import torch
from torchvision.models import resnet18, ResNet18_Weights

import torchvision.models as models
from torchvision.models import ResNet18_Weights
import torch
import torch.nn as nn

from efficientnet_pytorch import EfficientNet
model = EfficientNet.from_pretrained('efficientnet-b3')
num_classes = 10
model._fc = torch.nn.Linear(model._fc.in_features, num_classes)
# Initialisation des poids de la dernière couche
nn.init.kaiming_uniform_(model._fc.weight, nonlinearity='relu')

from sklearn.metrics import confusion_matrix
import seaborn as sns


import time

def acc(pred,label):
    pred = pred.argmax(dim=-1)  #prend L'INDICE de la valeur max parmi les predictions pour les 10 class, -1 sert a effectuer cette opération sur la dernière colonne de la matrice qui contient les prédictions par classe
    return torch.sum(pred == label).item() #ici label est sous forme d'indice donc juste une valeur scalaire

def test(args, data_loader, model):
    true = np.array([])
    pred = np.array([])
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    valid_loss = torch.tensor(0., device=device)
    valid_accuracy = torch.tensor(0., device=device)
    pbar = tqdm(data_loader)
    for i, (x, y) in enumerate(pbar):
        
        image = x.to(args.device)
        label = y.to(args.device)                

        output = model(image)

        loss = criterion(output, y)

        valid_loss += loss * data_loader.batch_size
        valid_accuracy += (torch.argmax(output, dim=1) == y).sum()
        
        label = label.squeeze()
        output = output.argmax(dim=-1)
        output = output.detach().cpu().numpy() #créer une nouvelle vue du tensor, .cpu deplace le tensor sur la mémoire CPU, .numpy() convertit le tensor en tableau Numpy
        pred = np.append(pred,output, axis=0) #
        
        label = label.detach().cpu().numpy()
        true =  np.append(true,label, axis=0)

        print(f'Valid loss: {valid_loss/len(valid_dataloader.dataset):.2f}')
        print(f'Valid accuracy: {100*valid_accuracy/len(valid_dataloader.dataset):.2f}')
    return valid_loss, valid_accuracy, pred, true


def train(args, data_loader, model):
    """
    TODO: Change the training code as you need. (e.g. different optimizer, different loss function, etc.)
            You can add validation code. -> This will increase the accuracy.
    """
    lr, weight_decay = 1e-5, 5e-4

    criterion = torch.nn.CrossEntropyLoss()
    params_1x = [param for name, param in model.named_parameters() if 'fc' not in str(name)]
    #optimizer = torch.optim.Adam([{'params':params_1x}, {'params': model.fc.parameters(), 'lr': lr*10}], lr=lr, weight_decay=args.weight_decay)    
    optimizer = torch.optim.Adam([{'params':params_1x}, {'params': model._fc.parameters(), 'lr': lr*10}], lr=lr)    
    
    epoch_times = []  # Liste pour stocker le temps par époque

    prediction = []
    labels = []

    for epoch in range(args.epochs):
        start_time = time.time()
        train_losses = [] 
        train_acc = 0.0
        total=0
        print(f"[Epoch {epoch+1} / {args.epochs}]")
        
        model.train()
        pbar = tqdm(data_loader)        #bar de chargement
        for i, (x, y) in enumerate(pbar):
            image = x.to(args.device)       #both line transfer matrix and label into the device(GPU in our case)
            label = y.to(args.device)          
            optimizer.zero_grad()

            output = model(image)
            
            label = label.squeeze()     #squeeze enleve les dimension de taille 1 de la matrice
            loss = criterion(output, label)
            prediction.append(output)
            labels.append(label)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item()) #convertit la matrice loss qui ne contient qu'un élément en un nombre scalaire
            total += label.size(0)   #total est le nombre total d'entrée(image), label.size donne la taille de la premiere dimension de la matrice label qui est généralement la batchsize

            train_acc += acc(output, label)

        epoch_train_loss = np.mean(train_losses)
        epoch_train_acc = train_acc/total


        print(f'Epoch {epoch+1}') 
        print(f'train_loss : {epoch_train_loss}')
        print('train_accuracy : {:.3f}'.format(epoch_train_acc*100))
      
        
        torch.save(model.state_dict(), f'{args.save_path}/model.pth')
        end_time = time.time()  # Fin du chronomètre pour l'époque
        epoch_duration = end_time - start_time
        epoch_times.append(epoch_duration)  # Enregistrement de la durée
    return train_acc, valid_accuracy, train_losses, valid_loss, epoch_times

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='2023 DL Term Project')
    parser.add_argument('--save-path', default='checkpoints/', help="Model's state_dict")
    parser.add_argument('--data', default='data/', type=str, help='data folder')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device
    
    """
    TODO: You can change the hyperparameters as you wish.
            (e.g. change epochs etc.)
    """
    
    # hyperparameters
    args.epochs = 5
    args.learning_rate = 0.01
    args.batch_size = 800
    #args.weight_decay = 5e-4

    # check settings
    print("==============================")
    print("Save path:", args.save_path)
    print('Using Device:', device)
    print('Number of usable GPUs:', torch.cuda.device_count())
    
    # Print Hyperparameter
    print("Batch_size:", args.batch_size)
    print("learning_rate:", args.learning_rate)
    print("Epochs:", args.epochs)
    #print("Weight_decay:", args.weight_decay)
    print("==============================")
    
    # Make Data loader and Model
    train_loader, valid_dataloader = make_data_loader(args)

    # custom model
    # model = BaseModel()
    # Charger le modèle pré-entraîné ResNet18
    '''model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    # Si vous voulez adapter le modèle à un nombre spécifique de classes
    num_classes = 10  # Remplacez par le nombre de vos classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Initialiser les poids de la nouvelle couche FC, si nécessaire
    nn.init.xavier_uniform_(model.fc.weight)'''

    # torchvision model
    #model = ResNet34CustomInput(num_classes, input_channels)
    
    # you have to change num_classes to 10
    #model.fc.out_features=10
    model.to(device)
    print(model)

    # Training The Model  # Données
    train_accuracy, train_loss, epoch_times = train(args, train_loader, model)
    val_accuracy, val_loss, pred, true = test(args, valid_dataloader, model)
    #Courbe de données
    import matplotlib.pyplot as plt

    epochs = args.epoch
    # Tracer l'accuracy
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_accuracy, label='Training Accuracy')
    plt.plot(epochs, val_accuracy, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    # Tracer la perte
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_loss, label='Training Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # Calcul du temps cumulé
    cumulative_time = np.cumsum(epoch_times)

    # Tracer la performance (par exemple, l'accuracy) en fonction du temps
    plt.plot(cumulative_time, train_accuracy, label='Train Accuracy')
    plt.plot(cumulative_time, val_accuracy, label='Validation Accuracy')
    plt.xlabel('Temps (s)')
    plt.ylabel('Accuracy')
    plt.title('Accuracy en Fonction du Temps')
    plt.legend()

    # Calculer la matrice de confusion
    conf_matrix = confusion_matrix(true, pred)

    # Tracer la matrice de confusion
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='g')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')

    plt.show()
