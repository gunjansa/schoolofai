import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torchsummary import summary

def create_mnist_dataloader():
    # Train data transformations
    # We train on complicated data by applying transformations to introduce Noise
    train_transforms = transforms.Compose([
        # This is applied to introduce some noise (like, Masking).
        # Example, if training is done to predict  Nurse from hospital teachers.
        # Then, model could be biased that by looking females it just predicts Nurse.
        transforms.RandomApply([transforms.CenterCrop(22), ], p=0.1),
        transforms.Resize((28, 28)),
        # There are chances that images are rotated. Like, if 7 is rotated it might predict as 1
        transforms.RandomRotation((-15., 15.), fill=0),
        transforms.ToTensor(),
        # In general we do normalize all the images with Mean and Standard Deviation. Why? Need to see previous lecture
        transforms.Normalize((0.1307,), (0.3081,)),
        ])

    # Test data transformations
    # In test data, we don't apply transformations
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1407,), (0.4081,))
        ])

    train_data = datasets.MNIST('../data', train=True, download=True, transform=train_transforms)
    # Changed
    # test_data = datasets.MNIST('../data', train=True, download=True, transform=train_transforms)
    test_data = datasets.MNIST('../data', train=False, download=True, transform=test_transforms)

    batch_size = 512

    kwargs = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 2, 'pin_memory': True}

    # test_loader = torch.utils.data.DataLoader(train_data, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_data, **kwargs)
    train_loader = torch.utils.data.DataLoader(train_data, **kwargs)
    return train_loader, test_loader

def get_dataset_images(train_loader, total_images=12):
    batch_data, batch_label = next(iter(train_loader))

    fig = plt.figure()

    for i in range(total_images):
        plt.subplot(3,4,i+1)
        plt.tight_layout()
        plt.imshow(batch_data[i].squeeze(0), cmap='gray')
        plt.title(batch_label[i].item())
        plt.xticks([])
        plt.yticks([])

def get_model_summary(model, device):
    return summary(model, input_size=(1, 28, 28))