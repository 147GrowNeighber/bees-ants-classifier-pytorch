import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import time
import os

from src.model import ResNet

CONFIG = {
    'DATA_DIR': './data/hymenoptera_data',
    'IMAGE_SIZE': 224,
    'BATCH_SIZE': 32,
    'LEARNING_RATE': 0.001,
    'MAX_EPOCHS': 25,
    'NUM_WORKERS': 0,
    'NUM_CLASSES': 2,
    'SAVE_MODEL_PATH': './models/best_model_improved.pth',
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CLASSES = ['ant', 'bee']


def get_data_transforms(phase):
    if phase == 'train':
        return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])


def create_data_loaders():
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(CONFIG['DATA_DIR'], x), get_data_transforms(x))
        for x in ['train', 'val']
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=CONFIG['BATCH_SIZE'], shuffle=(x == 'train'), num_workers=CONFIG['NUM_WORKERS'])
        for x in ['train', 'val']
    }

    return dataloaders, dataset_sizes


def train_model():
    since = time.time()

    dataloaders, dataset_sizes = create_data_loaders()
    print(f"Training set: {dataset_sizes['train']} images")
    print(f"Validation set: {dataset_sizes['val']} images")

    model = ResNet(CONFIG['NUM_CLASSES'])

    for param in model.parameters():
        param.requires_grad = False

    for param in model.backbone.net.layer4.parameters():
        param.requires_grad = True

    model = model.cuda() if torch.cuda.is_available() else model

    loss_criteria = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CONFIG['LEARNING_RATE']
    )
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    best_acc = 0.0

    if not os.path.exists('./models'):
        os.makedirs('./models')

    for epoch in range(CONFIG['MAX_EPOCHS']):
        print(f'\nEpoch {epoch+1}/{CONFIG["MAX_EPOCHS"]}')
        print('-' * 50)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                scheduler.step()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for images, labels in dataloaders[phase]:
                images = images.cuda() if torch.cuda.is_available() else images
                labels = labels.cuda() if torch.cuda.is_available() else labels

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(images)
                    _, preds = torch.max(outputs, 1)
                    loss = loss_criteria(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * images.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'[{phase.upper()}] Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save({
                    'model': model.state_dict(),
                    'best_accuracy': best_acc,
                    'classes': CLASSES,
                }, CONFIG['SAVE_MODEL_PATH'])
                print(f'  >> Saved best model, accuracy: {best_acc:.4f}')

    time_elapsed = time.time() - since
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best validation accuracy: {best_acc:.4f}')


if __name__ == '__main__':
    print('=' * 60)
    print('Bees vs Ants Classifier - Improved Training')
    print('=' * 60)
    train_model()
