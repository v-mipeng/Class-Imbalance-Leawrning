import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC as SVC

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical

from utils import *


DATA_DIR = "../data/gaussian/"



class MLPPolicy(nn.Module):
    '''
    Works when input dimension is low.
    '''
    def __init__(self, input_dim, hidden_dims, activations=None, output_dim=2):
        super(MLPPolicy, self).__init__()
        dims = [input_dim] + hidden_dims + [output_dim]
        self.linears = []
        for i in range(1, len(dims)):
            linear = nn.Linear(dims[i - 1], dims[i])
            setattr(self, 'linear_{}'.format(i), linear)
            self.linears.append(linear)
        if activations is not None:
            self.activations = activations
        else:
            self.activations = [nn.Sigmoid() for _ in range(len(dims)-2)] + [nn.Softmax(dim=1)]

    def forward(self, x):
        x_hat = x
        for linear, activation in zip(self.linears, self.activations):
            x_hat = activation(linear(x_hat))
        return x_hat


class HMLPPolicy(MLPPolicy):
    '''
    Works when input dimension is high.
    '''
    def __init__(self, input_dim, input_hidden_dims, union_hidden_dims, output_dim=2):
        super(MLPPolicy, self).__init__()
        input_dims = [input_dim] + input_hidden_dims
        union_dims = [input_hidden_dims[-1]+output_dim] + union_hidden_dims + [output_dim]
        self.input_linears = []
        for i in range(1, len(input_dims)):
            linear = nn.Linear(input_dims[i - 1], input_dims[i])
            setattr(self, 'linear_{}'.format(i), linear)
            self.input_linears.append(linear)
        self.union_linears = []
        for i in range(1, len(union_dims)):
            linear = nn.Linear(union_dims[i - 1], union_dims[i])
            setattr(self, 'linear_{}'.format(i+len(input_dims)), linear)
            self.union_linears.append(linear)
        self.input_activations = [nn.Sigmoid() for _ in range(len(input_dims) - 1)]
        self.union_activations = [nn.Sigmoid() for _ in range(len(union_dims) - 2)] + [nn.Softmax(dim=1)]

    def forward(self, x, y):
        x_hat = x
        for linear, activation in zip(self.input_linears, self.input_activations):
            x_hat = activation(linear(x_hat))
        x_hat = torch.cat([x_hat, y], dim=1)
        for linear, activation in zip(self.union_linears, self.union_activations):
            x_hat = activation(linear(x_hat))
        return x_hat


class GRUPolicy(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=2, activations=None):
        super(GRUPolicy, self).__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.output_dim = output_dim

    def forward(self, x):
        out, _ = self.gru(x.unsqueeze(0))
        out = out.view(out.size()[1], out.size(2))
        gru_out = out
        out = nn.Softmax(dim=1)(self.linear(out))
        return out, gru_out[-1]


class MLPValue(MLPPolicy):
    def __init__(self, input_dim, hidden_dims, activations=None, output_dim=1):
        super(MLPValue, self).__init__(input_dim, hidden_dims, activations=activations, output_dim=output_dim)

    def forward(self, x):
        x_hat = x
        for linear, activation in zip(self.linears, self.activations[:-1]):
            x_hat = activation(linear(x_hat))
        x_hat = self.linears[-1](x_hat)
        return x_hat


class ImbGaussianTrainer(object):
    def __init__(self):
        self.env = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5), random_state=1)

    def train(self, data_dir='../data/gaussian/', hidden_dims=None, major_ratio=0.05, lr=None):
        if hidden_dims is None:
            hidden_dims = [5]
        train_x, train_y, valid_x, valid_y, test_x, test_y = self.load_data(data_dir)
        self.policy = MLPPolicy(input_dim=train_x.shape[1] + train_y.shape[1], hidden_dims=hidden_dims)
        self.policy.cuda()
        if lr is None:
            self.optimizer = optim.RMSprop(self.policy.parameters())
        else:
            self.optimizer = optim.RMSprop(self.policy.parameters(), lr=lr)
        best_valid_reward = 0.
        best_test_reward = 0.
        i_episode = 10
        epoch = 0
        x = Variable(torch.cat([torch.from_numpy(train_x).float(), torch.from_numpy(train_y).float()], dim=1)).cuda()
        y = np.zeros((len(train_y), 2)).astype('float32')
        idx = np.argmax(train_y, axis=1)
        y[idx == 0] = [1-major_ratio, major_ratio]
        y[idx == 1] = [0., 1.]
        y = Variable(torch.from_numpy(y).cuda())
        self.initialize_policy(self.policy, x, y)
        while True:
            weight_probs = self.policy(x)
            cross_entropy = - torch.mean(torch.sum(weight_probs * torch.log(weight_probs + 1e-20), dim=1))
            self.reg = torch.mean(weight_probs[:, 1]) * 1e-4
            print(weight_probs)
            log_probs = []
            train_rewards = []
            valid_rewards = []
            test_rewards = []
            for i in range(i_episode):
                data_weights, log_prob = self.sample_weight(weight_probs)
                train_reward, valid_reward, test_reward = self.get_reward(train_x, train_y, data_weights,
                                                valid_x, valid_y, test_x, test_y)
                log_probs.append(log_prob)
                train_rewards.append(train_reward)
                valid_rewards.append(valid_reward)
                test_rewards.append(test_reward)
            if best_valid_reward < np.mean(valid_rewards):
                best_valid_reward = np.mean(valid_rewards)
                best_test_reward = np.mean(test_rewards)
            self.update_policy(log_probs, train_rewards)
            print('Train reward: {} in epoch: {} '.format(np.mean(train_rewards), epoch))
            print('Valid reward: {} in epoch: {} '.format(np.mean(valid_rewards), epoch))
            print('Test reward: {} in epoch: {} '.format(np.mean(test_rewards), epoch))
            print('Best valid F1: {}'.format(best_valid_reward))
            print('Best test F1: {}'.format(best_test_reward))
            epoch += 1
        print('Best valid F1: {}'.format(best_valid_reward))
        print('Best test F1: {}'.format(best_test_reward))

    def sample_weight(self, probs):
        if not isinstance(probs, Variable):
            probs = Variable(probs)
        m = Categorical(probs)
        action = m.sample()
        return action.data.cpu().numpy(), m.log_prob(action).mean().cuda()

    def get_reward(self, train_x, train_y, train_weights, valid_x, valid_y, test_x, test_y):
        '''Train the classifier with supervised

        :param train_x:
        :param train_y:
        :param train_weights:
        :param valid_x:
        :param valid_y:
        :return: The reward (F1)
        '''
        idx = train_weights == 1
        x = train_x[idx]
        y = train_y[idx]
        self.env.fit(x, y)
        preds = self.env.predict(train_x)
        _, _, train_reward = evaluate_f1(train_y, preds, pos_label=1)
        preds = self.env.predict(valid_x)
        _, _, valid_reward = evaluate_f1(valid_y, preds, pos_label=1)
        preds = self.env.predict(test_x)
        _, _, test_reward = evaluate_f1(test_y, preds, pos_label=1)
        return train_reward[1], valid_reward[1], test_reward[1]

    def update_policy(self, log_probs, rewards):
        rewards = Variable(torch.Tensor(rewards).cuda())
        policy_loss = []
        rewards = (rewards - rewards.mean()) / (rewards.std() + float(np.finfo(np.float32).eps))
        for log_prob, reward in zip(log_probs, rewards):
            policy_loss.append(-log_prob * reward)
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        print('policy loss:{}'.format(policy_loss.data.cpu()))
        print('reg:{}'.format(self.reg.data.cpu()))
        (policy_loss+self.reg).backward()
        self.optimizer.step()

    def load_data(self, data_dir):
        return load_imb_Gaussian(data_dir)

    def initialize_policy(self, policy, x, y, epoch=30):
        optimizer = optim.RMSprop(policy.parameters(), lr=0.001)
        for e in range(epoch):
            probs = policy(x)
            print(probs)
            loss = -torch.mean(torch.sum(y * torch.log(probs), dim=1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


class HImbGaussianTrainer(ImbGaussianTrainer):
    def train(self, data_dir='../data/gaussian/', input_hidden_dims=None, union_hidden_dims=None, major_ratio=0.1, lr=None):
        if input_hidden_dims is None:
            input_hidden_dims = [2]
        if union_hidden_dims is None:
            union_hidden_dims = [2]
        train_x, train_y, valid_x, valid_y, test_x, test_y = self.load_data(data_dir)
        self.policy = HMLPPolicy(train_x.shape[1], input_hidden_dims, union_hidden_dims).cuda()
        if lr is None:
            self.optimizer = optim.RMSprop(self.policy.parameters())
        else:
            self.optimizer = optim.RMSprop(self.policy.parameters(), lr=lr)
        best_valid_reward = 0.
        best_test_reward = 0.
        i_episode = 10
        epoch = 0
        x = Variable(torch.from_numpy(train_x).float().cuda())
        y = Variable(torch.from_numpy(train_y).float().cuda())
        y_hat = np.zeros((len(train_y), 2)).astype('float32')
        idx = np.argmax(train_y, axis=1)
        y_hat[idx == 0] = [1 - major_ratio, major_ratio]
        y_hat[idx == 1] = [0., 1.]
        y_hat = Variable(torch.from_numpy(y_hat).cuda())
        self.initialize_policy(self.policy, x, y, y_hat)
        while True:
            weight_probs = self.policy(x, y)
            self.reg = torch.mean(weight_probs[:, 1]) * 0
            print(weight_probs)
            log_probs = []
            train_rewards = []
            valid_rewards = []
            test_rewards = []
            for i in range(i_episode):
                data_weights, log_prob = self.sample_weight(weight_probs)
                train_reward, valid_reward, test_reward = self.get_reward(train_x, train_y, data_weights,
                                                                          valid_x, valid_y, test_x, test_y)
                log_probs.append(log_prob)
                train_rewards.append(train_reward)
                valid_rewards.append(valid_reward)
                test_rewards.append(test_reward)
            if best_valid_reward < np.mean(valid_rewards):
                best_valid_reward = np.mean(valid_rewards)
                best_test_reward = np.mean(test_rewards)
            self.update_policy(log_probs, train_rewards)
            print('Train reward: {} in epoch: {} '.format(np.mean(train_rewards), epoch))
            print('Valid reward: {} in epoch: {} '.format(np.mean(valid_rewards), epoch))
            print('Test reward: {} in epoch: {} '.format(np.mean(test_rewards), epoch))
            print('Best valid F1: {}'.format(best_valid_reward))
            print('Best test F1: {}'.format(best_test_reward))
            epoch += 1

    def initialize_policy(self, policy, x, y, y_hat, epoch=50):
        optimizer = optim.RMSprop(policy.parameters())
        for e in range(epoch):
            probs = policy(x, y)
            print(probs)
            loss = -torch.mean(torch.sum(y_hat * torch.log(probs), dim=1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


class GRUTrainer(ImbGaussianTrainer):
    def train(self, data_dir='../data/gaussian/', hidden_dim=50, major_ratio=0.05, lr=None):
        train_x, train_y, valid_x, valid_y, test_x, test_y = self.load_data(data_dir)
        self.policy = GRUPolicy(input_dim=train_x.shape[1] + train_y.shape[1], hidden_dim=hidden_dim)
        self.policy.cuda()
        if lr is None:
            self.optimizer = optim.RMSprop(self.policy.parameters())
        else:
            self.optimizer = optim.RMSprop(self.policy.parameters(), lr=lr)
        self.value = MLPValue(input_dim=hidden_dim, hidden_dims=[100])
        self.value.cuda()
        if lr is None:
            self.value_optimizer = optim.RMSprop(self.value.parameters())
        else:
            self.value_optimizer = optim.RMSprop(self.value.parameters(), lr=0.01)
        best_valid_reward = 0.
        best_test_reward = 0.
        best_train_reward = 0.
        i_episode = 10
        epoch = 0
        x = Variable(torch.cat([torch.from_numpy(train_x).float(), torch.from_numpy(train_y).float()], dim=1)).cuda()
        y = np.zeros((len(train_y), 2)).astype('float32')
        idx = np.argmax(train_y, axis=1)
        y[idx == 0] = [1 - major_ratio, major_ratio]
        y[idx == 1] = [0.1, 0.9]
        y = Variable(torch.from_numpy(y).cuda())
        self.initialize_policy(self.policy, x, y)
        self.epoch = epoch
        while True:
            weight_probs, s_T = self.policy(x)
            expected_reward = self.value(s_T)
            self.reg = torch.mean(weight_probs[:, 1]) ** 2 * 1e-3
            print(weight_probs[:5])
            log_probs = []
            train_rewards = []
            valid_rewards = []
            test_rewards = []
            for i in range(i_episode):
                data_weights, log_prob = self.sample_weight(weight_probs)
                train_reward, valid_reward, test_reward = self.get_reward(train_x, train_y, data_weights,
                                                                          valid_x, valid_y, test_x, test_y)
                log_probs.append(log_prob)
                train_rewards.append(train_reward)
                valid_rewards.append(valid_reward)
                test_rewards.append(test_reward)
            if best_train_reward < np.mean(train_rewards):
                best_valid_reward = np.mean(valid_rewards)
                best_test_reward = np.mean(test_rewards)
                best_train_reward = np.mean(train_rewards)
            self.update_policy(log_probs, train_rewards, Variable(expected_reward.data, requires_grad=False))
            if self.epoch % 10 == 0:
                self.update_value(expected_reward, np.mean(train_rewards))
            print('Train reward: {} in epoch: {} '.format(np.mean(train_rewards), self.epoch))
            print('Valid reward: {} in epoch: {} '.format(np.mean(valid_rewards), self.epoch))
            print('Test reward: {} in epoch: {} '.format(np.mean(test_rewards), self.epoch))
            print('Best train reward: {}'.format(best_train_reward))
            print('Best valid reward: {}'.format(best_valid_reward))
            print('Best test reward: {}'.format(best_test_reward))
            self.epoch += 1

    def update_value(self, expected_reward, reward):
        print(expected_reward)
        reward = Variable(torch.Tensor(np.array([reward])).cuda())
        self.value_optimizer.zero_grad()
        value_loss = (expected_reward-reward)**2
        print('value loss:{}'.format(value_loss.data.cpu()))
        (value_loss).backward(retain_graph=True)
        self.value_optimizer.step()

    def update_policy(self, log_probs, rewards, expected_reward):
        rewards = Variable(torch.Tensor(rewards).cuda())
        rewards -= expected_reward
        #if self.epoch > 20:
        #    rewards -= expected_reward
        #else:
        #    rewards = (rewards - rewards.mean()) / (rewards.std() + float(np.finfo(np.float32).eps))
        policy_loss = []
        for log_prob, reward in zip(log_probs, rewards):
            policy_loss.append(-log_prob * reward)
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        print('policy loss:{}'.format(policy_loss.data.cpu()))
        print('reg:{}'.format(self.reg.data.cpu()))
        (policy_loss + self.reg).backward(retain_graph =True)
        self.optimizer.step()

    def initialize_policy(self, policy, x, y, epoch=30):
        optimizer = optim.RMSprop(policy.parameters(), lr=0.001)
        for e in range(epoch):
            probs, state = policy(x)
            print(probs)
            loss = -torch.mean(torch.sum(y * torch.log(probs), dim=1))
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()


from sklearn import datasets, neighbors
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import StratifiedKFold

class LFWTrainer(GRUTrainer):
    '''Train on real data set (LFW)
    '''
    def __init__(self, folder):
        super(LFWTrainer, self).__init__()
        self.folder = folder
        self.env = neighbors.KNeighborsClassifier(3)
        self.cv = StratifiedKFold(n_splits=3)

    def load_data(self, data_dir):
        from sklearn.decomposition import PCA, TruncatedSVD
        from collections import Counter
        data = datasets.fetch_lfw_people()
        majority_person = 1871  # 530 photos of George W Bush
        minority_person = 531   # 29 photos of Bill Clinton
        majority_idxs = np.flatnonzero(data.target == majority_person)
        minority_idxs = np.flatnonzero(data.target == minority_person)
        idxs = np.hstack((majority_idxs, minority_idxs))
        x = data.data[idxs]
        y = data.target[idxs]
        y[y == majority_person] = 0
        y[y == minority_person] = 1
        train_x = TruncatedSVD(n_components=100).fit_transform(x)
        i = 1
        for train, test in self.cv.split(x, y):
            if i == self.folder:
                break
            i += 1
        #train_x = TruncatedSVD(n_components=100).fit_transform(x[train])
        y = np.eye(2)[y.astype('int32')]
        return train_x[train], y[train], x[train], y[train], x[test], y[test]

    def get_reward(self, train_x, train_y, train_weights, valid_x, valid_y, test_x, test_y):
        from imblearn.metrics import geometric_mean_score
        from sklearn.metrics import matthews_corrcoef
        idx = train_weights == 1
        x = valid_x[idx]
        y = valid_y[idx]
        self.env.fit(x, np.argmax(y, axis=1).astype('int32'))
        preds = self.env.predict_proba(test_x)
        if preds.shape[1] == 2:
            preds = preds[:, 1]
        valid_reward = evaluate_auc_roc(np.argmax(test_y, axis=1).astype('int32'), preds)

        return valid_reward, valid_reward, valid_reward


class CreditFraudTrainer(GRUTrainer):
    ''' 0.759 (785)
    Compared with: 0.70
    '''
    def __init__(self):
        super(CreditFraudTrainer, self).__init__()
        self.env = LogisticRegression(random_state=0, C=1e0)
        #self.env = MLPClassifier(solver='lbfgs', alpha=1., hidden_layer_sizes=(5), random_state=1234)
        #self.env = SVC(C=1e-1, random_state=0, dual=False)
        #self.env = DT(max_depth=7)

    def load_data(self, data_dir):
        x_train, y_train, x_valid, y_valid, x_test, y_test = load_imb_Credit_Fraud(data_dir)
        self.cost_mat_train = np.array([[1., 10., 0., 0.]]*len(x_train), dtype='float32')
        self.cost_mat_valid = np.array([[1., 10., 0., 0.]]*len(x_valid), dtype='float32')
        self.cost_mat_test = np.array([[1., 10., 0., 0.]]*len(x_test), dtype='float32')
        return x_train, y_train, x_valid, y_valid, x_test, y_test

    def get_reward(self, train_x, train_y, train_weights, valid_x, valid_y, test_x, test_y):
        idx = train_weights == 1
        x = train_x[idx]
        y = train_y[idx]
        self.env.fit(x, np.argmax(y, axis=1).astype('int32'))
        preds = self.env.predict_proba(train_x)[:, 1]
        train_reward = evaluate_auc_prc(np.argmax(train_y, axis=1).astype('int32'), preds, pos_label=1)
        preds = self.env.predict_proba(valid_x)[:, 1]
        valid_reward = evaluate_auc_prc(np.argmax(valid_y, axis=1).astype('int32'), preds, pos_label=1)
        #preds = self.env.predict_proba(test_x)[:, 1]
        #test_reward = evaluate_auc_prc(np.argmax(test_y, axis=1).astype('int32'), preds, pos_label=1)
        return train_reward, valid_reward, valid_reward


class PageTrainer(CreditFraudTrainer):
    '''0.88
    Compared with: 0.83
    '''
    def __init__(self):
        super(CreditFraudTrainer, self).__init__()
        #self.env = LogisticRegression(C=1.)
        self.env = MLPClassifier(solver='lbfgs', alpha=1e-1, hidden_layer_sizes=(5), random_state=1234)

    def load_data(self, data_dir):
        return load_imb_Page(data_dir)


from costcla.datasets import load_creditscoring2
from costcla.models import CostSensitiveLogisticRegression, ThresholdingOptimization
from costcla.metrics import savings_score, cost_loss
from sklearn.cross_validation import train_test_split
class CreditScoreTrainer(CreditFraudTrainer):
    def __init__(self):
        super(CreditScoreTrainer, self).__init__()
        #self.env = LogisticRegression(random_state=0)
        # self.env = MLPClassifier(solver='lbfgs', alpha=1., hidden_layer_sizes=(5), random_state=1234)
        #self.env = SVC(C=1e0, random_state=0)
        self.env = DT(max_depth=7)

    def load_data(self, data_dir):
        data = load_creditscoring2()
        cost_mat = data.cost_mat
        sets = train_test_split(data.data, data.target, cost_mat, test_size=0.5, random_state=0)
        x_train, x_test, y_train, y_test, cost_mat_train, cost_mat_test = sets
        self.cost_mat_train = cost_mat_train
        self.cost_mat_test = cost_mat_test
        self.cost_mat_valid = cost_mat_test
        return x_train, np.eye(2)[y_train], x_test, np.eye(2)[y_test], x_test, np.eye(2)[y_test]

    def get_reward(self, train_x, train_y, train_weights, valid_x, valid_y, test_x, test_y):
        idx = train_weights == 1
        x = train_x[idx]
        y = train_y[idx]
        self.env.fit(x, np.argmax(y, axis=1).astype('int32'))
        probs = self.env.predict_proba(train_x)
        preds = (probs[:, 1] > (self.cost_mat_train[:, 0] / self.cost_mat_train[:, 1])).astype('int32')
        train_reward = savings_score(np.argmax(train_y, axis=1).astype('int32'), preds, self.cost_mat_train)
        probs = self.env.predict_proba(valid_x)
        preds = (probs[:, 1] > (self.cost_mat_valid[:, 0] / self.cost_mat_valid[:, 1])).astype('int32')
        valid_reward = savings_score(np.argmax(valid_y, axis=1).astype('int32'), preds, self.cost_mat_valid)
        probs = self.env.predict_proba(test_x)
        preds = (probs[:, 1] > (self.cost_mat_test[:, 0] / self.cost_mat_test[:, 1])).astype('int32')
        test_reward = savings_score(np.argmax(test_y, axis=1).astype('int32'), preds, self.cost_mat_test)
        return train_reward, valid_reward, test_reward


from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.tree import DecisionTreeClassifier as DT
class SynTrainer(CreditFraudTrainer):
    def __init__(self):
        super(SynTrainer, self).__init__()
        self.env = LogisticRegression(C=1e1)
        #self.env = KNN(n_neighbors=10)
        #self.env = DT(max_depth=3)
        #self.env = SVC(C=1e3)

    def load_data(self, data_dir):
        return load_imb_Gaussian(data_dir)

    def get_reward(self, train_x, train_y, train_weights, valid_x, valid_y, test_x, test_y):
        idx = train_weights == 1
        x = train_x[idx]
        y = train_y[idx]
        self.env.fit(x, np.argmax(y, axis=1).astype('int32'))
        preds = self.env.predict(train_x)
        _, _, train_reward = evaluate_f1(np.argmax(train_y, axis=1).astype('int32'), preds, pos_label=1)
        preds = self.env.predict(valid_x)
        _, _, valid_reward = evaluate_f1(np.argmax(valid_y, axis=1).astype('int32'), preds, pos_label=1)
        preds = self.env.predict(test_x)
        _, _, test_reward = evaluate_f1(np.argmax(test_y, axis=1).astype('int32'), preds, pos_label=1)
        if self.epoch == 50:
            np.save('gaussian_weight.npy', np.array(train_weights))
        return train_reward[1], valid_reward[1], test_reward[1]


class CheckerBoardTrainer(CreditFraudTrainer):
    def __init__(self, imb_ratio=5):
        super(CheckerBoardTrainer, self).__init__()
        from sklearn.svm import SVC as SVM
        self.imb_ratio = imb_ratio
        self.env = SVM(C=1e4, kernel='rbf', random_state=0)

    def train(self, data_dir='../data/gaussian/', hidden_dim=50, major_ratio=0.05, lr=None):
        train_x, train_y, valid_x, valid_y, test_x, test_y = self.load_data(data_dir)
        self.policy = GRUPolicy(input_dim=train_x.shape[1] + train_y.shape[1], hidden_dim=hidden_dim)
        self.policy.cuda()
        if lr is None:
            self.optimizer = optim.RMSprop(self.policy.parameters())
        else:
            self.optimizer = optim.RMSprop(self.policy.parameters(), lr=lr)
        best_valid_reward = 0.
        best_test_reward = 0.
        best_train_reward = 0.
        i_episode = 10
        epoch = 0
        x = Variable(torch.cat([torch.from_numpy(train_x).float(), torch.from_numpy(train_y).float()], dim=1)).cuda()
        y = np.zeros((len(train_y), 2)).astype('float32')
        idx = np.argmax(train_y, axis=1)
        y[idx == 0] = [1 - major_ratio, major_ratio]
        y[idx == 1] = [0.1, 0.9]
        y = Variable(torch.from_numpy(y).cuda())
        self.initialize_policy(self.policy, x, y)
        self.epoch = epoch
        while True:
            weight_probs = self.policy(x)
            self.reg = torch.mean(weight_probs[:, 1]) ** 2 * 1e-3
            print(weight_probs[:5])
            log_probs = []
            train_rewards = []
            valid_rewards = []
            test_rewards = []
            for i in range(i_episode):
                data_weights, log_prob = self.sample_weight(weight_probs)
                train_reward, valid_reward, test_reward = self.get_reward(train_x, train_y, data_weights,
                                                                          valid_x, valid_y, test_x, test_y)
                log_probs.append(log_prob)
                train_rewards.append(train_reward)
                valid_rewards.append(valid_reward)
                test_rewards.append(test_reward)
            if best_train_reward < np.mean(train_rewards):
                best_valid_reward = np.mean(valid_rewards)
                best_test_reward = np.mean(test_rewards)
                best_train_reward = np.mean(train_rewards)
                np.save('weight_{}.npy'.format(self.imb_ratio), data_weights)
            self.update_policy(log_probs, train_rewards)
            print('Train reward: {} in epoch: {} '.format(np.mean(train_rewards), self.epoch))
            print('Valid reward: {} in epoch: {} '.format(np.mean(valid_rewards), self.epoch))
            print('Test reward: {} in epoch: {} '.format(np.mean(test_rewards), self.epoch))
            print('Best train reward: {}'.format(best_train_reward))
            print('Best valid reward: {}'.format(best_valid_reward))
            print('Best test reward: {}'.format(best_test_reward))
            self.epoch += 1

    def load_data(self, data_dir):
        train_x, train_y = load_checker_board(data_dir)
        return train_x, train_y, train_x, train_y, train_x, train_y

    def get_reward(self, train_x, train_y, train_weights, valid_x, valid_y, test_x, test_y):
        idx = train_weights == 1
        x = train_x[idx]
        y = train_y[idx]
        self.env.fit(x, np.argmax(y, axis=1).astype('int32'))
        preds = self.env.predict(train_x)
        train_reward = evaluate_macro_f1(np.argmax(train_y, axis=1).astype('int32'), preds, pos_label=1)
        return train_reward, train_reward, train_reward


class AmazonTrainer(object):
    def __init__(self):
        self.env = LogisticRegression(C=1) #MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5), random_state=1)

    def train(self, data_dir='../data/cmd/', hidden_dim=50, source_domain=0, target_domain=3, lr=None):

        train_x, train_y, valid_x, valid_y, test_x, test_y = self.load_data(data_dir,
                                                                            source_domain=source_domain,
                                                                            target_domain=target_domain)
        valid_x = valid_x[:100]
        valid_y = valid_y[:100]
        x = np.concatenate([train_x, valid_x])
        y = np.concatenate([train_y, valid_y])
        self.env.fit(x, np.argmax(y, axis=1))
        preds = self.env.predict(test_x)
        original_test_reward = accuracy_score(np.argmax(test_y, axis=1), preds)
        print('Original test reward:{}'.format(original_test_reward))
        dim = 30
        x = self.pca(train_x, dim=dim)
        self.policy = GRUPolicy(input_dim=dim + train_y.shape[1], hidden_dim=hidden_dim)
        self.policy.cuda()
        if lr is None:
            self.optimizer = optim.RMSprop(self.policy.parameters())
        else:
            self.optimizer = optim.RMSprop(self.policy.parameters(), lr=lr)
        best_valid_reward = 0.
        best_test_reward = 0.
        i_episode = 10
        epoch = 0
        x = Variable(torch.cat([torch.from_numpy(x).float(), torch.from_numpy(train_y).float()], dim=1)).cuda()
        while True:
            weight_probs = self.policy(x)
            self.reg = -torch.mean(weight_probs[:, 1]) * 1e-4
            print(weight_probs)
            log_probs = []
            train_rewards = []
            valid_rewards = []
            test_rewards = []
            for i in range(i_episode):
                data_weights, log_prob = self.sample_weight(weight_probs)
                train_reward, valid_reward, test_reward = self.get_reward(train_x, train_y, data_weights,
                                                                          valid_x, valid_y, test_x, test_y)
                log_probs.append(log_prob)
                train_rewards.append(train_reward)
                valid_rewards.append(valid_reward)
                test_rewards.append(test_reward)
            if best_valid_reward < np.mean(valid_rewards):
                best_valid_reward = np.mean(valid_rewards)
                best_test_reward = np.mean(test_rewards)
            self.update_policy(log_probs, valid_rewards)
            print('Train reward: {} in epoch: {} '.format(np.mean(train_rewards), epoch))
            print('Valid reward: {} in epoch: {} '.format(np.mean(valid_rewards), epoch))
            print('Test reward: {} in epoch: {} '.format(np.mean(test_rewards), epoch))
            print('Best valid reward: {}'.format(best_valid_reward))
            print('Best test reward: {}'.format(best_test_reward))
            print('Original test reward:{}'.format(original_test_reward))
            epoch += 1
        print('Best valid reward: {}'.format(best_valid_reward))
        print('Best test reward: {}'.format(best_test_reward))
        print('Original test reward:{}'.format(original_test_reward))

    def pca(self, x, dim=50):
        from sklearn.decomposition import PCA
        pca = PCA(n_components=dim)
        return pca.fit_transform(x)

    def sample_weight(self, probs):
        if not isinstance(probs, Variable):
            probs = Variable(probs)
        m = Categorical(probs)
        action = m.sample()
        return action.data.cpu().numpy(), m.log_prob(action).mean().cuda()

    def get_reward(self, train_x, train_y, train_weights, valid_x, valid_y, test_x, test_y):
        '''Train the classifier with supervised

        :param train_x:
        :param train_y:
        :param train_weights:
        :param valid_x:
        :param valid_y:
        :return: The reward (F1)
        '''
        from sklearn.metrics import accuracy_score
        idx = train_weights == 1
        x = train_x[idx]
        y = train_y[idx]
        self.env.fit(x, np.argmax(y, axis=1))
        preds = self.env.predict(train_x)
        train_reward = accuracy_score(np.argmax(train_y, axis=1), preds)
        preds = self.env.predict(valid_x)
        valid_reward = accuracy_score(np.argmax(valid_y, axis=1), preds)
        x = np.concatenate([x, valid_x])
        y = np.concatenate([y, valid_y])
        self.env.fit(x, np.argmax(y, axis=1))
        preds = self.env.predict(test_x)
        test_reward = accuracy_score(np.argmax(test_y, axis=1), preds)
        return train_reward, valid_reward, test_reward

    def update_policy(self, log_probs, rewards):
        rewards = Variable(torch.Tensor(rewards).cuda())
        policy_loss = []
        rewards = (rewards - rewards.mean()) / (rewards.std() + float(np.finfo(np.float32).eps))
        for log_prob, reward in zip(log_probs, rewards):
            policy_loss.append(-log_prob * reward)
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        print('policy loss:{}'.format(policy_loss.data.cpu()))
        print('reg:{}'.format(self.reg.data.cpu()))
        (policy_loss + self.reg).backward()
        self.optimizer.step()

    def load_data(self, data_dir, n_features=5000, source_domain=0, target_domain=3):
        return load_amazon(filename=data_dir, n_features=n_features,
                           source_domain=source_domain, target_domain=target_domain)

    def initialize_policy(self, policy, x, y, epoch=50):
        optimizer = optim.RMSprop(policy.parameters(), lr=0.001)
        for e in range(epoch):
            probs = policy(x)
            print(probs)
            loss = -torch.mean(torch.sum(y * torch.log(probs), dim=1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


class VehicleTrainer(GRUTrainer):
    def __init__(self):
        super(VehicleTrainer, self).__init__()
        from sklearn.svm import SVC as SVM
        #elf.env = MLPClassifier(solver='lbfgs', alpha=1e-3, hidden_layer_sizes=(8), random_state=1)
        #self.env = SVM(C=1e2, kernel='rbf', random_state=0) # For vehicle task
        #self.env = SVM(C=1e2, kernel='rbf', random_state=0, gamma=1e-2) # For page blocks
        #self.env = DT(max_depth=4) # For credit card task
        self.env = LogisticRegression(C=1e2, random_state=0) # For spam detection task

    def get_reward(self, train_x, train_y, train_weights, valid_x, valid_y, test_x, test_y):
        '''Train the classifier with supervised

        :param train_x:
        :param train_y:
        :param train_weights:
        :param valid_x:
        :param valid_y:
        :return: The reward (F1)
        '''
        from imblearn.metrics import geometric_mean_score
        from sklearn.metrics import matthews_corrcoef
        idx = train_weights == 1
        x = train_x[idx]
        y = train_y[idx]
        self.env.fit(x, np.argmax(y, axis=1).astype('int32'))
        preds = self.env.predict(valid_x)
        valid_reward = evaluate_f2(np.argmax(valid_y, axis=1).astype('int32'), preds)
        return valid_reward, valid_reward, valid_reward

    def load_data(self, data_dir):
        train_x, train_y, valid_x, valid_y = load_imb_Vehicle(data_dir)
        return train_x, train_y, valid_x, valid_y, valid_x, valid_y



if __name__ == '__main__':
    #trainer = HImbGaussianTrainer()
    #trainer.train(input_hidden_dims=[5], union_hidden_dims=[])
    #trainer = SynTrainer()
    #trainer.train(hidden_dims=[10, 5])
    #trainer = LFWTrainer()
    #trainer.train(hidden_dim=100, major_ratio=0.2, lr=0.001)
    #trainer = CreditFraudTrainer()
    #trainer.train(data_dir='../data/real/creditcard/', hidden_dim=50, major_ratio=0.1, lr=0.001)
    #trainer = PageTrainer()
    #trainer.train(data_dir='../data/real/page/', hidden_dim=25, major_ratio=0.1, lr=0.001)
    #trainer = SynTrainer()
    #trainer.train(data_dir='../data/gaussian/', hidden_dim=25, major_ratio=0.1, lr=0.0001)
    #trainer = CreditScoreTrainer()
    #trainer.train(data_dir='../data/real/creditcard/', hidden_dim=50, major_ratio=0.1, lr=0.001)
    #trainer = AmazonTrainer()
    #trainer.train(data_dir='../data/real/cmd/amazon.mat', hidden_dim=50, source_domain=2, target_domain=1, lr=0.001)
    #import sys
    import sys
    name = sys.argv[1]
    trainer = VehicleTrainer()
    trainer.train(data_dir='../data/real/spam/train.pkl'.format(name), hidden_dim=50, major_ratio=0.1, lr=0.001)
    #imb_ratio = int(sys.argv[1])
    #trainer = CheckerBoardTrainer(imb_ratio=imb_ratio)
    #trainer.train(data_dir='../data/synthetic/checkerboard/train_{}.pkl'.format(imb_ratio), hidden_dim=25,
    #              major_ratio=1./imb_ratio, lr=0.001)
    #folder = int(sys.argv[1])
    #trainer = LFWTrainer(folder=folder)
    #trainer.train(data_dir='..', hidden_dim=25, major_ratio=0.2, lr=0.0002)


