import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.optim as optim


use_cuda = torch.cuda.is_available()


class MLPNet(nn.Module):
    def __init__(self):
        super(MLPNet, self).__init__()
        self.fc1 = nn.Linear(28*28, 500)
        self.fc2 = nn.Linear(500, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def name(self):
        return "MLP"


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def name(self):
        return "LeNet"


class MNISTTrainer(object):
    def __init__(batch_size=100, model=LeNet()):
        self.batch_size=batch_size
        self.model = model

    def fit(train_x, train_y):
        dataset = torch.utils.data.TensorDataset(torch.from_numpy(trian_x), torch.from_numpy(train_y))
        train_loader = torch.utils.data.DataLoader(
                         dataset=dataset,
                         batch_size=self.batch_size,
                         shuffle=True)
        if use_cuda:
            self.model = self.model.cuda()
        optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        ceriation = nn.CrossEntropyLoss()

        for epoch in xrange(10):
            # trainning
            ave_loss = 0
            for batch_idx, (x, target) in enumerate(train_loader):
                optimizer.zero_grad()
                if use_cuda:
                    x, target = x.cuda(), target.cuda()
                x, target = Variable(x), Variable(target)
                out = self.model(x)
                loss = ceriation(out, target)
                loss.backward()
                optimizer.step()

    def predict(test_x):
        preds = []
        dataset = torch.utils.data.TensorDataset(torch.from_numpy(test_x), torch.from_numpy(np.ones(len(test_x))))
        train_loader = torch.utils.data.DataLoader(
                         dataset=dataset,
                         batch_size=batch_size,
                         shuffle=False)
        for batch_idx, x in enumerate(test_loader):
            if use_cuda:
                x = x.cuda()
            x = Variable(x, volatile=True)
            out = self.model(x)
            _, pred_label = torch.max(out.data, 1)
            preds += pred_labels.tolist()
        return np.array(preds)


class MLP(nn.Module):
    def __init__(self, gamma=0.001):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, 1)
        #self.fc4 = nn.Linear(6, 1)
        self.gamma = gamma

    def forward(self, x):
        x = self.fc1(x)**2 #torch.exp(self.gamma*self.fc1(x)) #torch.tanh(self.fc1(x))
        x = self.fc2(x)**2
        x = self.fc3(x)
        #x = self.fc4(x)
        return x

    def name(self):
        return "MLP"


def rbf(x, weights, gamma):
    """
    Applies a rbf transformation to the incoming data: :math:y = exp(-gamma*(w[None, :, :]-x[:, None, :])^2).

    Shape:
        - x: :math:`(N, in\_features)` where `*` means any number of
          additional dimensions
        - Weights: :math:`(out\_features, in\_features)`
        - Output: :math:`(N, out\_features)`
    """
    y = (weights.unsqueeze(0) - x.unsqueeze(1))**2
    y = y.sum(dim=-1)
    return torch.exp(-gamma * y)


class RBFMLP(nn.Module):
    def __init__(self, gamma=None):
        super(RBFMLP, self).__init__()
        self.fc1 = nn.Linear(11, 6)
        self.fc2 = nn.Linear(6, 1)
        if gamma is None:
            gamma = 1.0 / 10.
        self.gamma = gamma

    def forward(self, x):
        x = rbf(x, self.fc1.weight, self.gamma)
        x = self.fc2(x)
        return x

    def name(self):
        return "RBFMLP"


class MLPRegression(object):
    def __init__(self, gamma=None, batch_size=1000):
        self.batch_size = batch_size
        self.model = MLP(gamma)

    def fit(self, train_x, train_y):
        dataset = torch.utils.data.TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
        train_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=True)
        if use_cuda:
            self.model = self.model.cuda()
        optimizer = optim.RMSprop(self.model.parameters(), lr=0.0001)
        ceriation = nn.MSELoss()

        for epoch in range(1000):
            for batch_idx, (x, target) in enumerate(train_loader):
                optimizer.zero_grad()
                if use_cuda:
                    x, target = x.cuda(), target.float().cuda()
                x, target = Variable(x), Variable(target)
                out = self.model(x)
                loss = ceriation(out, target)
                loss.backward()
                optimizer.step()
                print(loss.data[0])
        return self

    def predict(self, test_x):
        preds = []
        dataset = torch.utils.data.TensorDataset(torch.from_numpy(test_x), torch.from_numpy(np.ones(len(test_x))))
        test_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=False)
        for batch_idx, (x, y) in enumerate(test_loader):
            if use_cuda:
                x = x.cuda()
            x = Variable(x, volatile=True)
            out = self.model(x)
            preds += out.data.cpu().numpy().tolist()
        return np.array(preds)


if __name__ == '__main__':
    import numpy as np
    np.random.seed = 123
    x = 5*np.random.rand(100000, 10).astype('float32')
    #x = np.concatenate([x, np.ones((len(x), 1), dtype='float32')], axis=1)
    y = (x[:, 0] * x[:, 8]**3).astype('float32')
    np.random.seed = 1234
    x_test = 5*np.random.rand(100000, 10).astype('float32')+5
    #x_test = np.concatenate([x_test, np.ones((len(x_test), 1), dtype='float32')], axis=1)
    y_test = (x_test[:, 0] * x_test[:, 8]**3).astype('float32')
    from sklearn.neural_network import MLPRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error
    from sklearn.svm import SVR
    #rg = MLPRegressor(hidden_layer_sizes=(50, ), activation='tanh', tol=1e-7)
    rg = MLPRegression(gamma=1.)
    preds = rg.fit(x, y).predict(x)
    mse = mean_squared_error(y, preds)
    print('train mse:{}'.format(mse))
    preds = rg.predict(x_test)
    mse = mean_squared_error(y_test, preds)
    print('test mse:{}'.format(mse))

