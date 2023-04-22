import torch
from torch import nn, optim
from sklearn import metrics
from torch.utils.data import TensorDataset, DataLoader


def train_full_batch(model_input, model, target_output, learning_rate, max_iter, gpu=False):
    """
    train the model
    :param model_input:
    :param model:
    :param target_output:
    :param learning_rate:
    :param max_iter:
    :param gpu: gpu or not, {Ture, False(default)}
    :return: loss in each iteration
    """
    if gpu:
        device_gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # go to gpu
        model.to(device_gpu)
        model_input, target_output = model_input.to(device_gpu), target_output.to(device_gpu)

    model_input = model_input.double()
    target_output = target_output.double()

    # define loss and optimizer
    criterion = mse_loss_fun
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # 迭代更新
    loss_his = []  # 准备记录训练过程中的loss
    for iter_th in range(max_iter):
        # compute model outputs
        model_output = model.forward(model_input)

        # compute loss
        loss = criterion(model_output, target_output)
        loss_his.append(loss.data)

        # iterate
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print
        print('{}-th iteration, training loss: {:.4f}'.format(iter_th, loss.data))

    if gpu:
        # back to cpu
        device_cpu = torch.device("cpu")
        model.to(device_cpu)

    return torch.tensor(loss_his)


def test(model_input, model, target_output, task='classification', pre_sca=None):
    """
    test the model
    :param model_input:
    :param model:
    :param target_output:
    :param task: tsak type, {classification(default),regression}
    :param pre_sca: preprocessing scaler for target outputs
            only task='regression'
    :return: performance metric, loss or acc
    """
    model_input = model_input.double()
    target_output = target_output.double()

    # model outputs on test samples
    model_output = model.forward(model_input)

    # define loss fun
    criterion = mse_loss_fun

    if task == 'classification':
        # classification, compute loss and acc
        loss = criterion(model_output, target_output)
        acc = metrics.accuracy_score(target_output.argmax(dim=1), model_output.argmax(dim=1))
        return loss, acc
    elif task == 'regression':
        # regression, de-normalized model outputs, only compute loss
        model_output = torch.DoubleTensor(pre_sca.inverse_transform(model_output.data))
        loss = criterion(model_output, target_output)
        return loss


def mse_loss_fun(y: torch.tensor = None, z: torch.tensor = None):
    """
    self-defined mse loss
    :param y: model outputs
    :param z: target outputs
    :return: mse loss
    """
    return ((y - z) ** 2).sum() / (2 * y.size(0))
