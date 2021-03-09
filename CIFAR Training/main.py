import torch
from torchvision.models import resnet18
import torch_pruning as tp
from Pruning import *
import matplotlib.pyplot as plt
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device ' + str(device))

list_models = [[2,2,2]]
print("les modèles prunés sont: " + str(list_models))

courbe = []
size = []
miles = [5,8,15]
for i in list_models:
    model = ResNet18(i)
    model.to(device=device)
    print("modèle: " + str(i))
    PATH = str(i) + ".pth"
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint)
    model.to(device=device)
    criterion = nn.CrossEntropyLoss()
    dim = 0
    lim = 0
    courbe += [evaluation(model, test_loader, criterion, device)]
    size += [print_nonzeros(model)]
    for j in range(15):
        if j == miles[lim]:
            lim += 1
        print("étape numéro " + str(j) + " du prunage")
        pruner(model, dim, lim, 0.1)
        dim += 1
        dim %= 2
        optimizer = torch.optim.SGD(model.parameters(), 0.1, weight_decay=0.0005)
        training(40, train_loader, valid_loader, model, criterion, optimizer, 0.1, device)
        courbe += [evaluation(model, test_loader, criterion, device)]
        size += [print_nonzeros(model)]
    #clean(model)
    optimizer = torch.optim.SGD(model.parameters(), 0.01, weight_decay=0.0005)
    training(80, train_loader, valid_loader, model, criterion, optimizer, 0.1, device)
    evaluation(model, test_loader, criterion, device)
    torch.save(model, "model_more_structured" + str(i) + ".pth")

print(courbe)
print(size)

""""
PATH = '[1,1,1,1]/model[1,1,1,1].pth'
checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint)
model.to(device=device)
criterion = nn.CrossEntropyLoss()
courbe = []
size = []
courbe += [evaluation(model, test_loader, criterion, device)]
print_nonzeros(model)
size += [print_nonzeros(model)]
dim = 0
for name, module in model.named_modules():
        # prune 20% of connections in all 2D-conv layers
    if isinstance(module, torch.nn.Conv2d):
            prune.ln_structured(module, name='weight', amount=0.65, n=2, dim=dim)
dim += 1
dim %= 2
optimizer = torch.optim.SGD(model.parameters(), 0.1, weight_decay=0.0005)
training(100, train_loader, valid_loader, model, criterion, optimizer, 0.1, device)
courbe += [evaluation(model, test_loader, criterion, device)]
print_nonzeros(model)
size += [print_nonzeros(model)]

clean(model)
evaluation(model, test_loader, criterion, device)
torch.save(model.state_dict(), 'modeldirect70%[1,1,1,1].pth')

model.load_state_dict(checkpoint)
model.to(device=device)
criterion = nn.CrossEntropyLoss()
courbe = []
size = []
courbe += [evaluation(model, test_loader, criterion, device)]
print_nonzeros(model)
size += [print_nonzeros(model)]
dim = 0
for i in range(3):
    for name, module in model.named_modules():
        # prune 20% of connections in all 2D-conv layers
        if isinstance(module, torch.nn.Conv2d):
            prune.ln_structured(module, name='weight', amount=0.3, n=2, dim=dim)
    dim += 1
    dim %= 2
    optimizer = torch.optim.SGD(model.parameters(), 0.1)
    courbe += [evaluation(model, test_loader, criterion, device)]
    print_nonzeros(model)
    size += [print_nonzeros(model)]

optimizer = torch.optim.SGD(model.parameters(), 0.1, weight_decay=0.0005)
training(100, train_loader, valid_loader, model, criterion, optimizer, 0.1, device)
clean(model)
evaluation(model, test_loader, criterion, device)
torch.save(model.state_dict(), 'modeletapenotrain70%[1,1,1,1].pth')

model.load_state_dict(checkpoint)
model.to(device=device)
criterion = nn.CrossEntropyLoss()
courbe = []
size = []
courbe += [evaluation(model, test_loader, criterion, device)]
print_nonzeros(model)
size += [print_nonzeros(model)]
dim = 0
for i in range(3):
    for name, module in model.named_modules():
        # prune 20% of connections in all 2D-conv layers
        if isinstance(module, torch.nn.Conv2d):
            prune.ln_structured(module, name='weight', amount=0.3, n=2, dim=dim)
    dim += 1
    dim %= 2
    optimizer = torch.optim.SGD(model.parameters(), 0.1, weight_decay=0.0005)
    training(100, train_loader, valid_loader, model, criterion, optimizer, 0.1, device)
    courbe += [evaluation(model, test_loader, criterion, device)]
    print_nonzeros(model)
    size += [print_nonzeros(model)]

clean(model)
evaluation(model, test_loader, criterion, device)
torch.save(model.state_dict(), 'modeletapetrain70%[1,1,1,1].pth')

model.load_state_dict(checkpoint)
model.to(device=device)
criterion = nn.CrossEntropyLoss()
courbe = []
size = []
courbe += [evaluation(model, test_loader, criterion, device)]
print_nonzeros(model)
size += [print_nonzeros(model)]
dim = 0
for i in range(5):
    for name, module in model.named_modules():
        # prune 20% of connections in all 2D-conv layers
        if isinstance(module, torch.nn.Conv2d):
            prune.ln_structured(module, name='weight', amount=0.2, n=2, dim=dim)
    dim += 1
    dim %= 2
    optimizer = torch.optim.SGD(model.parameters(), 0.1, weight_decay=0.0005)
    training(100, train_loader, valid_loader, model, criterion, optimizer, 0.1, device)
    courbe += [evaluation(model, test_loader, criterion, device)]
    print_nonzeros(model)
    size += [print_nonzeros(model)]

clean(model)
evaluation(model, test_loader, criterion, device)
torch.save(model.state_dict(), 'modelpetiteetapetrain70%[1,1,1,1].pth')

print(courbe)
print(size)
plt.figure()
plt.ylabel('Accuracy')
plt.plot(size,courbe)
plt.xlabel('% of parameters pruned')
plt.show()

"""