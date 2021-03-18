import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data.dataloader import DataLoader
import time


def training(n_epochs, train_loader, valid_loader, model, criterion, optimizer, factor, device, bina=True,
             valid_loss_min=np.Inf, milestone=[25, 35]):  # FUNCTION TO BE COMPLETED

    lamda = factor
    train_losses, valid_losses = [], []
    torch.autograd.set_detect_anomaly(True)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, verbose=True, milestones=milestone, gamma=0.1)

    if bina:
        for epoch in range(n_epochs):
            train_loss, valid_loss = 0, 0

            model.train()
            for data, label in train_loader:
                data = data.to(device=device, dtype=torch.float32)
                label = label.to(device=device, dtype=torch.long)
                optimizer.zero_grad()  # clear the gradients of all optimized variables
                output = model(data)  # forward pass: compute predicted outputs by passing inputs to the model
                loss = criterion(output, label)  # calculate the loss
                loss.backward()  # backward pass: compute gradient of the loss with respect to model parameters
                optimizer.step()  # perform a single optimization step (parameter update)
                train_loss += loss.item() * data.size(0)  # update running training loss

            model.eval()
            for data, label in valid_loader:
                data = data.to(device=device, dtype=torch.float32)
                label = label.to(device=device, dtype=torch.long)
                with torch.no_grad():
                    output = model(data)
                loss = criterion(output, label)
                valid_loss += loss.item() * data.size(0)

            train_loss /= len(train_loader.sampler)
            valid_loss /= len(valid_loader.sampler)
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            lr_scheduler.step()
            print(
                'epoch: {} \ttraining Loss: {:.6f} \tvalidation Loss: {:.6f}'.format(epoch + 1, train_loss, valid_loss))

            if valid_loss <= valid_loss_min:
                print('validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                    valid_loss_min,
                    valid_loss))
                torch.save(model.state_dict(), 'model.pt')
                valid_loss_min = valid_loss
    else:
        for epoch in range(n_epochs):
            train_loss, valid_loss = 0, 0

            model.model.train()
            for data, label in train_loader:
                model.binarization()
                data = data.to(device=device, dtype=torch.float32)
                label = label.to(device=device, dtype=torch.long)
                optimizer.zero_grad()  # clear the gradients of all optimized variables
                output = model.forward(data)  # forward pass: compute predicted outputs by passing inputs to the model
                loss = criterion(output, label)  # calculate the loss
                loss.backward()  # backward pass: compute gradient of the loss with respect to model parameters
                model.restore()
                optimizer.step()  # perform a single optimization step (parameter update)
                model.clip()
                train_loss += loss.item() * data.size(0)  # update running training loss

            model.model.eval()
            for data, label in valid_loader:
                data = data.to(device=device, dtype=torch.float32)
                label = label.to(device=device, dtype=torch.long)
                with torch.no_grad():
                    output = model.model(data)
                loss = criterion(output, label)
                valid_loss += loss.item() * data.size(0)

            train_loss /= len(train_loader.sampler)
            valid_loss /= len(valid_loader.sampler)
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            lr_scheduler.step(valid_loss)
            print(
                'epoch: {} \ttraining Loss: {:.6f} \tvalidation Loss: {:.6f}'.format(epoch + 1, train_loss, valid_loss))

            if valid_loss <= valid_loss_min:
                print('validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                    valid_loss_min,
                    valid_loss))
                torch.save(model.model.state_dict(), 'model.pt')
                valid_loss_min = valid_loss

    return train_losses, valid_losses, valid_loss_min


def evaluation(model, test_loader, criterion, device):
    test_loss = 0.0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    model.eval()
    for data, label in test_loader:
        data = data.to(device=device, dtype=torch.float32)
        label = label.to(device=device, dtype=torch.long)
        with torch.no_grad():
            output = model(data)
        loss = criterion(output, label)
        test_loss += loss.item() * data.size(0)
        _, pred = torch.max(output, 1)
        correct = np.squeeze(pred.eq(label.data.view_as(pred)))
        for i in range(len(label)):
            digit = label.data[i]
            class_correct[digit] += correct[i].item()
            class_total[digit] += 1

    test_loss = test_loss / len(test_loader.sampler)
    print('test Loss: {:.6f}\n'.format(test_loss))
    for i in range(10):
        print('test accuracy of %d: %2d%% (%2d/%2d)' % (
        i, 100 * class_correct[i] / class_total[i], np.sum(class_correct[i]), np.sum(class_total[i])))
    print('\ntest accuracy (overall): %2.2f%% (%2d/%2d)' % (
    100. * np.sum(class_correct) / np.sum(class_total), np.sum(class_correct), np.sum(class_total)))
    return (100. * np.sum(class_correct) / np.sum(class_total))


def plot_loss(test_loss, training_loss, n_epochs):
    plt.figure()
    test_ax = np.arange(0, n_epochs, n_epochs / len(test_loss))
    train_ax = np.arange(0, n_epochs, n_epochs / len(training_loss))

    plt.plot(test_ax, test_loss, "r")
    plt.plot(train_ax, training_loss)


def multiple_trainings(factors, epochs, pre_trained=False):
    total_losses = []
    if pre_trained == False:
        for f in factors:
            model = ResNet18()
            model.to(device=device)

            criterion = nn.CrossEntropyLoss()
            trainloader = DataLoader(minicifar_train, batch_size=32, sampler=train_sampler)
            validloader = DataLoader(minicifar_train, batch_size=32, sampler=valid_sampler)
            testloader = DataLoader(minicifar_test, batch_size=32)
            optimizer = torch.optim.SGD(model.parameters(), 0.005)

            train_losses, valid_losses = training(epochs, trainloader, validloader, model, criterion, optimizer, f)
            total_losses.append([f, train_losses, valid_losses])

            PATH = 'model_untrained' + str(f) + '.pth'
            print(PATH)
            torch.save(model.state_dict(), PATH)
            print(PATH + " has been saved")

    if pre_trained == True:
        for f in factors:
            model = my_Model(1000)
            model.to(device=device)
            freeze_resnet(model)

            criterion = nn.CrossEntropyLoss()
            trainloader = DataLoader(minicifar_train_im, batch_size=32, sampler=train_sampler)
            validloader = DataLoader(minicifar_train_im, batch_size=32, sampler=valid_sampler)
            testloader = DataLoader(minicifar_test_im, batch_size=32)
            optimizer = torch.optim.SGD(model.parameters(), 0.005)

            train_losses, valid_losses = training(epochs, trainloader, validloader, model, criterion, optimizer, f)
            total_losses.append([f, train_losses, valid_losses])

            PATH = 'model_pretrained' + str(f) + '.pth'
            torch.save(model.state_dict(), PATH)

    return total_losses


def lr_trainings(learning_rates, epochs, pre_trained=False):
    total_losses = []
    if pre_trained == False:
        for lr in learning_rates:
            model = ResNet18()
            model.to(device=device)

            criterion = nn.CrossEntropyLoss()
            trainloader = DataLoader(minicifar_train, batch_size=32, sampler=train_sampler)
            validloader = DataLoader(minicifar_train, batch_size=32, sampler=valid_sampler)
            testloader = DataLoader(minicifar_test, batch_size=32)
            optimizer = torch.optim.SGD(model.parameters(), lr)

            train_losses, valid_losses = training(epochs, trainloader, validloader, model, criterion, optimizer, 0.25)
            total_losses.append([lr, train_losses, valid_losses])

            PATH = 'lr_model_untrained' + str(lr) + '.pth'
            print(PATH)
            torch.save(model.state_dict(), PATH)
            print(PATH + " has been saved")

    if pre_trained == True:
        for lr in learning_rates:
            model = my_Model(1000)
            model.to(device=device)
            freeze_resnet(model)

            criterion = nn.CrossEntropyLoss()
            trainloader = DataLoader(minicifar_train_im, batch_size=32, sampler=train_sampler)
            validloader = DataLoader(minicifar_train_im, batch_size=32, sampler=valid_sampler)
            testloader = DataLoader(minicifar_test_im, batch_size=32)
            optimizer = torch.optim.SGD(model.parameters(), lr)

            train_losses, valid_losses = training(epochs, trainloader, validloader, model, criterion, optimizer, 0.25)
            total_losses.append([lr, train_losses, valid_losses])

            PATH = 'lr_model_pretrained' + str(lr) + '.pth'
            torch.save(model.state_dict(), PATH)

    return total_losses


def create_model(lr, epoch, bina=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device ' + str(device))

    model_1 = models.resnet18(pretrained=True)
    model_1.fc = nn.Linear(512, 4)
    model_1.to(device=device)

    if bina:
        model = binnaryconnect.BC(model_1)

    criterion = nn.CrossEntropyLoss()

    trainloader = DataLoader(minicifar_train_im, batch_size=32, sampler=train_sampler)
    validloader = DataLoader(minicifar_train_im, batch_size=32, sampler=valid_sampler)
    testloader = DataLoader(minicifar_test_im, batch_size=32)

    optimizer = torch.optim.SGD(model_1.parameters(), lr)

    start_time = time.time()

    train_losses_1, valid_losses_1 = training(epoch, trainloader, validloader, model_1, criterion, optimizer, 0.1,
                                              device)

    print("--- %s seconds ---" % (time.time() - start_time))
    t = (time.time() - start_time) / epoch * 10
    PATH = 'data/model.pth'
    torch.save(model_1.state_dict(), PATH)

    evaluation(model_1, testloader, criterion, device)
    return model_1, t


def training_distillation(n_epochs, train_loader, valid_loader, model, teacher, criterion1, criterion2, optimizer,
                          factor, device, bina=True, valid_loss_min=np.Inf,
                          milestone=[25, 35]):  # FUNCTION TO BE COMPLETED

    lamda = factor
    train_losses, valid_losses = [], []
    torch.autograd.set_detect_anomaly(True)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, verbose=True, milestones=milestone, gamma=0.1)

    if bina:
        for epoch in range(n_epochs):
            train_loss, valid_loss = 0, 0

            model.train()
            for data, label in train_loader:
                data = data.to(device=device, dtype=torch.float32)
                label = label.to(device=device, dtype=torch.long)
                optimizer.zero_grad()  # clear the gradients of all optimized variables
                output = model(data)  # forward pass: compute predicted outputs by passing inputs to the model
                output_teacher = teacher(data)
                loss = criterion1(output, label) + criterion2(output, output_teacher)  # calculate the loss
                #loss = criterion2(output, output_teacher)
                #loss = criterion1(output, label)
                loss.backward()  # backward pass: compute gradient of the loss with respect to model parameters
                optimizer.step()  # perform a single optimization step (parameter update)
                train_loss += loss.item() * data.size(0)  # update running training loss

            model.eval()
            for data, label in valid_loader:
                data = data.to(device=device, dtype=torch.float32)
                label = label.to(device=device, dtype=torch.long)
                with torch.no_grad():
                    output = model(data)
                loss = criterion1(output, label)
                valid_loss += loss.item() * data.size(0)

            train_loss /= len(train_loader.sampler)
            valid_loss /= len(valid_loader.sampler)
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            lr_scheduler.step()
            print(
                'epoch: {} \ttraining Loss: {:.6f} \tvalidation Loss: {:.6f}'.format(epoch + 1, train_loss, valid_loss))

            if valid_loss <= valid_loss_min:
                print('validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                    valid_loss_min,
                    valid_loss))
                torch.save(model.state_dict(), 'model.pt')
                valid_loss_min = valid_loss
    else:
        for epoch in range(n_epochs):
            train_loss, valid_loss = 0, 0

            model.model.train()
            for data, label in train_loader:
                model.binarization()
                data = data.to(device=device, dtype=torch.float32)
                label = label.to(device=device, dtype=torch.long)
                optimizer.zero_grad()  # clear the gradients of all optimized variables
                output = model.forward(data)  # forward pass: compute predicted outputs by passing inputs to the model
                loss = criterion(output, label)  # calculate the loss
                loss.backward()  # backward pass: compute gradient of the loss with respect to model parameters
                model.restore()
                optimizer.step()  # perform a single optimization step (parameter update)
                model.clip()
                train_loss += loss.item() * data.size(0)  # update running training loss

            model.model.eval()
            for data, label in valid_loader:
                data = data.to(device=device, dtype=torch.float32)
                label = label.to(device=device, dtype=torch.long)
                with torch.no_grad():
                    output = model.model(data)
                loss = criterion(output, label)
                valid_loss += loss.item() * data.size(0)

            train_loss /= len(train_loader.sampler)
            valid_loss /= len(valid_loader.sampler)
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            lr_scheduler.step(valid_loss)
            print(
                'epoch: {} \ttraining Loss: {:.6f} \tvalidation Loss: {:.6f}'.format(epoch + 1, train_loss, valid_loss))

            if valid_loss <= valid_loss_min:
                print('validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                    valid_loss_min,
                    valid_loss))
                torch.save(model.model.state_dict(), 'model.pt')
                valid_loss_min = valid_loss

    return train_losses, valid_losses, valid_loss_min


def evaluation_distillation(model, teacher, test_loader, criterion1, criterion2, device):
    test_loss = 0.0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    model.eval()
    for data, label in test_loader:
        data = data.to(device=device, dtype=torch.float32)
        label = label.to(device=device, dtype=torch.long)
        with torch.no_grad():
            output = model(data)
            output_teacher = teacher(data)
        loss = criterion1(output, label) + criterion2(output, output_teacher)
        test_loss += loss.item() * data.size(0)
        _, pred = torch.max(output, 1)
        correct = np.squeeze(pred.eq(label.data.view_as(pred)))
        for i in range(len(label)):
            digit = label.data[i]
            class_correct[digit] += correct[i].item()
            class_total[digit] += 1

    test_loss = test_loss / len(test_loader.sampler)
    print('test Loss: {:.6f}\n'.format(test_loss))
    for i in range(10):
        print('test accuracy of %d: %2d%% (%2d/%2d)' % (
        i, 100 * class_correct[i] / class_total[i], np.sum(class_correct[i]), np.sum(class_total[i])))
    print('\ntest accuracy (overall): %2.2f%% (%2d/%2d)' % (
    100. * np.sum(class_correct) / np.sum(class_total), np.sum(class_correct), np.sum(class_total)))
    return (100. * np.sum(class_correct) / np.sum(class_total))
