import torch
from torchvision.models import resnet18
import torch_pruning as tp
from Pruning import *
import matplotlib.pyplot as plt
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device ' + str(device))


PATH = "densenet_trained.pth"
model = torch.load(PATH)
model.to(device=device)


courbe = []
size = []
criterion = nn.CrossEntropyLoss()

courbe += [evaluation(model, test_loader, criterion, device)]
size += [print_nonzeros(model)]

miles = [5,10,15,20]
lim=0
dim = 0
for j in range(20):
    if j == miles[lim]:
        lim += 1
    print("étape numéro " + str(j) + " du prunage")

    pruner100(model, dim,lim, 0, 0.1)
    criterion = nn.CrossEntropyLoss()
    dim += 1
    dim %= 2
    optimizer = torch.optim.SGD(model.parameters(), 0.1, weight_decay=0.0005)
    training(40, train_loader, valid_loader, model, criterion, optimizer, 0.1, device)
    courbe += [evaluation(model, test_loader, criterion, device)]
    size += [print_nonzeros(model)]
    torch.save(model, "densenet_pruned_more_structured_etape_" + str(j) + ".pth")

#clean(model)
optimizer = torch.optim.SGD(model.parameters(), 0.01, weight_decay=0.0005)
training(80, train_loader, valid_loader, model, criterion, optimizer, 0.1, device, milestone=[40,60])
evaluation(model, test_loader, criterion, device)
torch.save(model, "densenet_pruned_more_structured.pth")


"""
list_models = [[2,2,2]]
print("les modèles prunés sont: " + str(list_models))

courbe = []
size = []
miles = [5,8,15]
for i in list_models:
    print("modèle: " + str(i))
    PATH = "model_trained" + str(i) + ".pth"
    #PATH = str(i) + ".pth"
    model = torch.load(PATH)
    model.to(device=device)
    optimizer = torch.optim.SGD(model.parameters(), 0.1, weight_decay=0.0005)
    criterion = nn.CrossEntropyLoss()
    training(100, train_loader, valid_loader, model, criterion, optimizer, 0.1, device, milestones = [50,80])
    torch.save(model, "model_trained" + str(i) + ".pth")
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