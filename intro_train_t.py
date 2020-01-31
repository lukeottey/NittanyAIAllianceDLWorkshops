import argparse
import torch

import ml_models
import datasets

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='mnist', \
        choices=['cifar10', 'cifar100', 'svhn', 'stl10', 'mnist', 'fashion_mnist', 'kmnist', 'emnist', 'qmnist'])
    parser.add_argument('-m', '--model', type=str, default='SoftmaxRegression_t', choices=['LogisticRegression_t', 'SoftmaxRegression_t'])
    parser.add_argument('-lf', '--loss_fn', type=str, default='nll', choices=['mse', 'nll'])
    parser.add_argument('-lr', '--learning_rate', dest='lr', type=float, default=0.01)
    parser.add_argument('-mt', '--momentum', type=float, default=0.9, help='only used if optimizer uses momentum')
    parser.add_argument('-b', '--batch_size', type=int, default=32)
    parser.add_argument('-e', '--num_epochs', type=int, default=40)
    parser.add_argument('-wd', '--weight_decay', type=float, default=5e-4)
    parser.add_argument('-ss', '--subset', type=int, nargs='+', default=[-1])
    args = parser.parse_args()
    return args

def setup(args):
    loss_fn_ = {
        'mse': lambda: torch.nn.MSELoss(),
        'nll': lambda: torch.nn.functional.nll_loss
        }[args.loss_fn]()
    if args.loss_fn == 'nll' and args.model == 'SoftmaxRegression_t':
        loss_fn = lambda y_hat, y: loss_fn_(y_hat.log(), y)
    else:
        loss_fn = loss_fn_
    
    subset, num_classes = {
        True: (args.subset, len(args.subset)),
        False: (None, datasets.num_classes_lookup_dict[args.dataset])
        }[args.model in ('LogisticRegression_t', 'SoftmaxRegression_t') and sum(args.subset) != -1]
    
    trainloader, testloader = datasets.__dict__[args.dataset](root='./data', batch_size=args.batch_size, subset=subset, return_datasets=False)
    for x, _ in trainloader:
        num_features = x.size(-1) * x.size(-2) * x.size(-3)
        break
    model = ml_models.__dict__[args.model](num_features, num_classes)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
    return model, optimizer, trainloader, testloader, loss_fn

def trainer(model, optimizer, loader, loss_fn, enable_grad=True):
    running_loss = 0.
    correct, total = 0, 0
    with torch.set_grad_enabled(enable_grad):
        for batch_idx, (X, y) in enumerate(loader):
            X = X.view(X.size(0), -1).type(torch.float32)
            y_hat = model(X)
            loss = loss_fn(y_hat, y)
            if enable_grad:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            _, pred = y_hat.max(1)
            correct += pred.eq(y).to('cpu', copy=False).sum().item()
            total += X.size(0)
            running_loss += loss.item()
    return correct / total, running_loss / batch_idx

def main():
    args = get_args()
    model, optimizer, trainloader, testloader, loss_fn = setup(args)
    best_acc = -1
    for epoch in range(1, args.num_epochs + 1):
        train_acc, train_loss = trainer(model, optimizer, trainloader, loss_fn, enable_grad=True)
        test_acc, test_loss = trainer(model, None, testloader, loss_fn, enable_grad=False)
        print("\nEpoch [{}/{}]: Train Accuracy: {} | Train Loss: {} | Test Accuracy: {} | Test Loss: {}".format(epoch, args.num_epochs, \
            round(train_acc, 4), round(train_loss, 4), round(test_acc, 4), round(test_loss, 4)))
        if test_acc > best_acc:
            best_acc = test_acc
        print("Best test accuracy = {}\n".format(round(best_acc, 4)))
        if epoch in args.milestones:
            model.adjust_lr(gamma=args.gamma)


if __name__ == '__main__':
    main()
