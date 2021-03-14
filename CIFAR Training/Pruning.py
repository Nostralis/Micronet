from ResNet import *
from datasetGen import *
from trainer import *
import torch.nn.utils.prune as prune


def pruner(model, dim, lim, pruning_rate):
    for name, module in model.named_modules():
        # prune 20% of connections in all 2D-conv layers
        if name != "conv1" and len(name)>0:

            layer = name.split(".")[0]
            lay = layer[:-1]
            nb = layer[-1]
            if lay == 'layer' and int(nb) > lim:
                if isinstance(module, torch.nn.Conv2d):
                    prune.ln_structured(module, name='weight', amount=pruning_rate, n=2, dim=dim)

    return model

def pruner100(model, dim, limb_lock, lim_lay, pruning_rate):
    for name, module in model.named_modules():
        # prune 20% of connections in all 2D-conv layers
        if len(name) > 0:
            layer = name.split(".")
            if len(layer) > 4:
                if layer[2][:-1] == "denseblock":
                    if layer[3][:-1] == "denselayer":
                        block = int(layer[2][-1])
                        lay = int(layer[3][-1])
                        if block > lim_block:
                            if lay > lim_lay:
                                if isinstance(module, torch.nn.Conv2d):
                                    prune.ln_structured(module, name='weight', amount=pruning_rate, n=2, dim=dim)


def clean(model):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.remove(module, name='weight')

    return model


def countZeroWeights(model):
    zeros = 0
    for param in model.named_buffers():
        if param is not None:
            zeros += torch.sum((param == 0)).data.item()
    return zeros


def prun(pru):
    trainloader = DataLoader(minicifar_train_im, batch_size=32, sampler=train_sampler)
    validloader = DataLoader(minicifar_train_im, batch_size=32, sampler=valid_sampler)
    testloader = DataLoader(minicifar_test_im, batch_size=32)

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
    for i in range(pru):
        pruner(model, dim)
        dim += 1
        dim %= 2
        optimizer = torch.optim.SGD(model.parameters(), 0.1)
        training(30, train_loader, valid_loader, model, criterion, optimizer, 0.1, device)
        courbe += [evaluation(model, test_loader, criterion, device)]
        print_nonzeros(model)
        size += [print_nonzeros(model)]

    clean(model)
    evaluation(model, test_loader, criterion, device)
    torch.save(model.state_dict(), 'data/model222.pth')
    return model, courbe, size


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    print(pp)


def print_nonzeros(model):
    nonzero = total = 0
    for name, p in model.named_buffers():
        tensor = p.data.cpu().numpy()
        nz_count = np.count_nonzero(tensor)
        total_params = np.prod(tensor.shape)
        nonzero += nz_count
        total += total_params
        print(f'{name:20} | nonzeros = {nz_count:7} / {total_params:7} ({100 * nz_count / total_params:6.2f}%) | total_pruned = {total_params - nz_count :7} | shape = {tensor.shape}')
    print(f'alive: {nonzero}, pruned : {total - nonzero}, total: {total}, Compression rate : {total/nonzero:10.2f}x  ({100 * (total-nonzero) / total:6.2f}% pruned)')
    return 100 * (total-nonzero) / total
