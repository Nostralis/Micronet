import torch
from torchvision.models import resnet18
import torch_pruning as tp
from Pruning import *
import matplotlib.pyplot as plt
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device ' + str(device))

student_ = [1, 1, 1]
teacher_ = [2, 2, 2, 2]

student = ResNet18(student_)
teacher = ResNet18(teacher_)
model = torch.load(str(teacher_) + ".pth")
teacher.load_state_dict(model)

student.to(device)
teacher.to(device)


criterion = nn.CrossEntropyLoss()


def teacher_loss(output, label):
    return torch.mean((output - label) ** 2)


optimizer = torch.optim.SGD(student.parameters(), 0.1, weight_decay=0.0005)

training_distillation(150, train_loader, valid_loader, student, teacher, criterion, teacher_loss, optimizer, 0.1, device,
                      milestone=[80, 130])

evaluation_distillation(student, teacher, test_loader, criterion, teacher_loss, device)


PATH = "model_distill_[2,2,2].pth"
torch.save(student,PATH)