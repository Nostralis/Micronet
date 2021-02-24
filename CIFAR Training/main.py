from Pruning import *
import matplotlib.pyplot as plt
import time


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device ' + str(device))
model = ResNet18()

PATH = 'model_classic_2.pth'
checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint)
model.to(device=device)

"""criterion = nn.CrossEntropyLoss()
evaluation(model,test_loader,criterion,device)
optimizer = torch.optim.SGD(model.parameters(), 0.5)
training(20, train_loader, valid_loader, model, criterion, optimizer, 0.1, device)
PATH = 'data/modelcp1.pth'
torch.save(model.state_dict(), PATH)
evaluation(model,test_loader,criterion,device)
optimizer = torch.optim.SGD(model.parameters(), 0.05)
training(20, train_loader, valid_loader, model, criterion, optimizer, 0.1, device)
PATH = 'data/modelcp2.pth'
torch.save(model.state_dict(), PATH)
evaluation(model,test_loader,criterion,device)
optimizer = torch.optim.SGD(model.parameters(), 0.005)
training(20, train_loader, valid_loader, model, criterion, optimizer, 0.1, device)
evaluation(model,test_loader,criterion,device)

PATH = 'data/modelcp3.pth'
torch.save(model.state_dict(), PATH)"""

model,courbe,size,timeC=prun(6, 60)
print(courbe)
print(size)
plt.figure()
plt.ylabel('Accuracy')
plt.plot(size,courbe)
plt.xlabel('% of parameters pruned')
plt.show()