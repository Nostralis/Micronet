import torch
from torchvision.models import resnet18
import torch_pruning as tp
from Pruning import *
import matplotlib.pyplot as plt
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device ' + str(device))

model1_ = torch.load("[1, 1, 1, 1].pth")
model1 = ResNet18([4])
#model1.load_state_dict(model1_)
model1.to(device)

model2_ = torch.load("[2, 2, 2, 2].pth")
model2 = ResNet18([1,1])
#model2.load_state_dict(model2_)
model2.to(device)

model3_ = torch.load("[1, 1, 1].pth")
model3 = ResNet18([2,2])
#model3.load_state_dict(model3_)
model3.to(device)

#model4 = torch.load("model_trained[2, 2, 2].pth")
model4 = ResNet18([2,2,2])
#model4.load_state_dict(model4_)
model4.to(device)

criterion = nn.CrossEntropyLoss()

optimizer1 = torch.optim.SGD(model1.parameters(), 0.1, weight_decay=0.0005)
optimizer2 = torch.optim.SGD(model2.parameters(), 0.1, weight_decay=0.0005)
optimizer3 = torch.optim.SGD(model3.parameters(), 0.1, weight_decay=0.0005)
optimizer4 = torch.optim.SGD(model4.parameters(), 0.1, weight_decay=0.0005)

print("model [4]")
training(150, train_loader, valid_loader, model1, criterion, optimizer1, 0.1, device, milestone=[80, 130])
evaluation(model1, test_loader, criterion, device)

print("model [1,1]")
training(150, train_loader, valid_loader, model2, criterion, optimizer2, 0.1, device, milestone=[80, 130])
evaluation(model2, test_loader, criterion, device)

print("model [2,2]")
training(150, train_loader, valid_loader, model3, criterion, optimizer3, 0.1, device, milestone=[80, 130])
evaluation(model3, test_loader, criterion, device)

print("model [2,2,2]")
training(150, train_loader, valid_loader, model4, criterion, optimizer4, 0.1, device, milestone=[80, 130])
evaluation(model4, test_loader, criterion, device)



""""
pruner(model1, 0, 0, 0)
print("model 1,1,1,1")
print_nonzeros(model1)
evaluation(model1, test_loader, criterion, device)

pruner(model2, 0, 0, 0)
print("model 2,2,2,2")
print_nonzeros(model2)
evaluation(model2, test_loader, criterion, device)

pruner(model3, 0, 0, 0)
print("1,1,1")
print_nonzeros(model3)
evaluation(model3, test_loader, criterion, device)

pruner(model4, 0, 0, 0)
print("2,2,2")
print_nonzeros(model4)
evaluation(model4, test_loader, criterion, device)


student_ = [1, 1]
teacher_ = [2, 2, 2, 2]

student = torch.load("model_distill_[1,1].pth")
teacher = ResNet18(teacher_)
model = torch.load(str(teacher_) + ".pth")
teacher.load_state_dict(model)
student.to(device)
teacher.to(device)












criterion = nn.CrossEntropyLoss()

def teacher_loss(output, label):
    return torch.mean((output - label) ** 2)


optimizer = torch.optim.SGD(student.parameters(), 0.1, weight_decay=0.0005)

training_distillation(150, train_loader, valid_loader, student, teacher, criterion, teacher_loss, optimizer, 0.1,
                      device,
                      milestone=[80, 130])

evaluation_distillation(student, teacher, test_loader, criterion, teacher_loss, device)

PATH = "model_distill_[1,1]V2.pth"
torch.save(student, PATH)

"""
