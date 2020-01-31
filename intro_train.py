import argparse

import ml_models
import datasets

def get_args(): # python3 intro_train.py -d mnist -b 32 -o GD -lr 0.001
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='mnist', choices=['mnist', 'svhn'])
    parser.add_argument('-m', '--model', type=str, default='SoftmaxRegression', choices=['LogisticRegression', 'SoftmaxRegression'])
    parser.add_argument('-lf', '--loss_fn', type=str, default='nll', choices=['mse', 'nll'])
    parser.add_argument('-o', '--optimizer', type=str, default='GD', choices=['GD', 'Momentum'])
    parser.add_argument('-lr', '--learning_rate', dest='lr', type=float, default=0.01)
    parser.add_argument('-mi', '--milestones', type=int, nargs='+', default=[-1])
    parser.add_argument('-g', '--gamma', type=float, default=0.2)
    parser.add_argument('-mt', '--momentum', type=float, default=0.9, help='only used if optimizer uses momentum')
    parser.add_argument('-b', '--batch_size', type=int, default=128)
    parser.add_argument('-e', '--num_epochs', type=int, default=40)
    parser.add_argument('-lm', '--lambd', type=float, default=0.1, help='regularization constant')
    parser.add_argument('-r', '--regularization', type=str, default='L1', choices=['L1', 'L2', 'Elastic', None])
    args = parser.parse_args()
    return args

def trainer(model, loader, enable_grad=True):
    running_acc, running_loss = 0, 0.
    for batch_idx, (X, y) in enumerate(loader):
        X = X.view(X.size(0), -1).numpy()
        y = y.numpy()
        if enable_grad:
            acc, loss = model.fit(X, y)
        else:
            acc, loss = model.evaluate(X, y)
        running_acc += acc
        running_loss += loss
    return running_acc / batch_idx, running_loss / batch_idx

def main():
    args = get_args()
    regularizer = {
        'L1': lambda: ml_models.L1Regularizer(lambd=args.lambd),
        'L2': lambda: ml_models.L2Regularizer(lambd=args.lambd),
        'Elastic': lambda: ml_models.ElasticRegularizer(lambd=args.lambd, ratio=0.5),
        None: lambda: None
    }[args.regularization]()
    optimizer = {
        'GD': lambda: ml_models.GradientDescent(lr=args.lr, regularizer=regularizer),
        'Momentum': lambda: ml_models.Momentum(lr=args.lr, momentum=args.momentum, regularizer=regularizer)}[args.optimizer]()
    model = ml_models.__dict__[args.model](optimizer=optimizer, loss=ml_models.__dict__[args.loss_fn])
    trainloader, testloader = datasets.__dict__[args.dataset](root='./data', batch_size=args.batch_size, return_datasets=False)
    for x, y in trainloader:
        x = x.view(x.size(0), -1).numpy()
        num_features = x.shape[-1]
        break
    model.initialize(num_features=num_features)
    best_acc = -1
    for epoch in range(1, args.num_epochs + 1):
        train_acc, train_loss = trainer(model, trainloader, enable_grad=True)
        test_acc, test_loss = trainer(model, testloader, enable_grad=False)
        print("\nEpoch [{}/{}]: Train Accuracy: {} | Train Loss: {} | Test Accuracy: {} | Test Loss: {}".format(epoch, args.num_epochs, \
            round(train_acc, 4), round(train_loss, 4), round(test_acc, 4), round(test_loss, 4)))
        if test_acc > best_acc:
            best_acc = test_acc
        print("Best test accuracy = {}\n".format(round(best_acc, 4)))
        if epoch in args.milestones:
            model.adjust_lr(gamma=args.gamma)


if __name__ == '__main__':
    main()
