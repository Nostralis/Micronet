from Pruning import *
import matplotlib.pyplot as plt
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ResNet18()
model.to(device=device)

print('Using device ' + str(device))
PATH = '[1,1,1]/model[1,1,1] (2).pth'
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
    optimizer = torch.optim.SGD(model.parameters(), 0.1)
    training(100, train_loader, valid_loader, model, criterion, optimizer, 0.1, device)
    courbe += [evaluation(model, test_loader, criterion, device)]
    print_nonzeros(model)
    size += [print_nonzeros(model)]

clean(model)
evaluation(model, test_loader, criterion, device)
torch.save(model.state_dict(), 'modeldirect70%.pth')

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

optimizer = torch.optim.SGD(model.parameters(), 0.1)
training(100, train_loader, valid_loader, model, criterion, optimizer, 0.1, device)
clean(model)
evaluation(model, test_loader, criterion, device)
torch.save(model.state_dict(), 'modeletapenotrain70%.pth')

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
    training(100, train_loader, valid_loader, model, criterion, optimizer, 0.1, device)
    courbe += [evaluation(model, test_loader, criterion, device)]
    print_nonzeros(model)
    size += [print_nonzeros(model)]

clean(model)
evaluation(model, test_loader, criterion, device)
torch.save(model.state_dict(), 'modeletapetrain70%.pth')

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
    optimizer = torch.optim.SGD(model.parameters(), 0.1)
    training(100, train_loader, valid_loader, model, criterion, optimizer, 0.1, device)
    courbe += [evaluation(model, test_loader, criterion, device)]
    print_nonzeros(model)
    size += [print_nonzeros(model)]

clean(model)
evaluation(model, test_loader, criterion, device)
torch.save(model.state_dict(), 'modelpetiteetapetrain70%.pth')

print(courbe)
print(size)
plt.figure()
plt.ylabel('Accuracy')
plt.plot(size,courbe)
plt.xlabel('% of parameters pruned')
plt.show()