import torchvision.transforms as transforms
import numpy as np
from torchvision.datasets import CIFAR10
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torch.utils.data.sampler import SubsetRandomSampler


n_classes_minicifar = 4
train_size = 0.8
R = 5

# Download the entire CIFAR10 dataset


## Normalization is different when training from scratch and when training using an imagenet pretrained backbone

normalize_scratch = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

normalize_forimagenet = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# Data augmentation is needed in order to train from scratch
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize_scratch,
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    normalize_scratch,
])

## No data augmentation when using Transfer Learning
## however resize to Imagenet input dimensions is recommended for Transfer learning
transform_train_imagenet = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    normalize_forimagenet,
])

transform_test_imagenet = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    normalize_forimagenet,
])



### The data from CIFAR10 will be downloaded in the following dataset
rootdir = './data/cifar100'

c10train = CIFAR10(rootdir, train=True, download=True, transform=transform_train)
c10test = CIFAR10(rootdir, train=False, download=True, transform=transform_test)

c10train_imagenet = CIFAR10(rootdir, train=True, download=True, transform=transform_train_imagenet)
c10test_imagenet = CIFAR10(rootdir, train=False, download=True, transform=transform_test_imagenet)


trainset = CIFAR10(root='./data', train=True,
                                        download=True, transform=transform_train)

testset = CIFAR10(root='./data', train=False,
                                       download=True, transform=transform_test)

# Generating Mini-CIFAR
#
# CIFAR10 is sufficiently large so that training a model up to the state of the art performance will take approximately 3 hours on the 1060 GPU available on your machine.
# As a result, we will create a "MiniCifar" dataset, based on CIFAR10, with less classes and exemples.

def train_validation_split(train_size, num_train_examples):
    # obtain training indices that will be used for validation
    indices = list(range(num_train_examples))
    np.random.shuffle(indices)
    idx_split = int(np.floor(train_size * num_train_examples))
    train_index, valid_index = indices[:idx_split], indices[idx_split:]

    # define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_index)
    valid_sampler = SubsetRandomSampler(valid_index)

    return train_sampler, valid_sampler


def generate_subset(dataset, n_classes, reducefactor, n_ex_class_init):
    nb_examples_per_class = int(np.floor(n_ex_class_init / reducefactor))
    # Generate the indices. They are the same for each class, could easily be modified to have different ones. But be careful to keep the random seed!

    indices_split = np.random.RandomState(seed=42).choice(n_ex_class_init, nb_examples_per_class, replace=False)

    all_indices = []
    for curclas in range(n_classes):
        curtargets = np.where(np.array(dataset.targets) == curclas)
        indices_curclas = curtargets[0]
        indices_subset = indices_curclas[indices_split]
        # print(len(indices_subset))
        all_indices.append(indices_subset)
    all_indices = np.hstack(all_indices)

    return Subset(dataset, indices=all_indices)



batch_size = 512
valid_size = 0.2


def create_data_loaders(batch_size, valid_size, train_data, test_data):  # FUNCTION TO BE COMPLETED

    test_loader = DataLoader(test_data, batch_size=batch_size)

    num_train, num_test = len(train_data), len(test_data)
    # obtain training indices that will be used for validation
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_index, valid_index = indices[split:], indices[:split]

    # define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_index)
    valid_sampler = SubsetRandomSampler(valid_index)

    # prepare data loaders
    train_loader = DataLoader(train_data, batch_size=batch_size, sampler=train_sampler)
    valid_loader = DataLoader(train_data, batch_size=batch_size, sampler=valid_sampler)

    return train_loader, valid_loader, test_loader

train_loader, valid_loader, test_loader = create_data_loaders(batch_size, valid_size, trainset, testset)

### These dataloader are ready to be used to train for scratch
minicifar_train = generate_subset(dataset=c10train, n_classes=n_classes_minicifar, reducefactor=R, n_ex_class_init=5000)
num_train_examples = len(minicifar_train)
train_sampler, valid_sampler = train_validation_split(train_size, num_train_examples)
minicifar_test = generate_subset(dataset=c10test, n_classes=n_classes_minicifar, reducefactor=1, n_ex_class_init=1000)

### These dataloader are ready to be used to train using Transfer Learning
### from a backbone pretrained on ImageNet
minicifar_train_im = generate_subset(dataset=c10train_imagenet, n_classes=n_classes_minicifar, reducefactor=R,
                                     n_ex_class_init=5000)
num_train_examples_im = len(minicifar_train_im)
train_sampler_im, valid_sampler_im = train_validation_split(train_size, num_train_examples_im)
minicifar_test_im = generate_subset(dataset=c10test_imagenet, n_classes=n_classes_minicifar, reducefactor=1,n_ex_class_init=1000)
