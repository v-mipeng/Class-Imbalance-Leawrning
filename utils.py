import os
import pickle

import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import normalize

from sklearn.metrics import (f1_score, accuracy_score, precision_recall_fscore_support,
                             precision_recall_curve, auc, roc_curve)
from imblearn.metrics import classification_report_imbalanced


def evaluate_f1(y, y_pred, pos_label=1):
    precision, recall, f1, support = precision_recall_fscore_support(y, y_pred, pos_label=pos_label)
    return precision, recall, f1


def evaluate_macro_f1(y, y_pred, pos_label=1):
    f1 = f1_score(y, y_pred, pos_label=pos_label, average='macro')
    return f1


def evaluate_auc_prc(y, pred):
    precision, recall, thresholds = precision_recall_curve(y, pred)
    aucprc = auc(recall, precision)
    return aucprc


def evaluate_auc_roc(y, pred):
    fpr, tpr, thresholds = roc_curve(y, pred)
    roc_auc = auc(fpr, tpr)
    return roc_auc


def evaluate_f2(y, y_pred):
    precision, recall, f1, support = precision_recall_fscore_support(y, y_pred, pos_label=1)
    #print classification_report(y, y_pred)
    f2 = (1+0.5**2)*(precision[1]*recall[1])/(0.5**2*precision[1]+recall[1])
    return f2


def load_imb_Gaussian(data_dir):
    train = pickle.load(open(os.path.join(data_dir, 'train.pkl'), 'rb'), encoding='bytes')
    train_x, train_y = train[b'x'], train[b'y']
    valid = pickle.load(open(os.path.join(data_dir, 'valid.pkl'), 'rb'), encoding='bytes')
    valid_x, valid_y = valid[b'x'], valid[b'y']
    test = pickle.load(open(os.path.join(data_dir, 'test.pkl'), 'rb'), encoding='bytes')
    test_x, test_y = test[b'x'], test[b'y']
    return train_x, train_y, valid_x, valid_y, test_x, test_y


def load_imb_Credit_Fraud(data_dir):
    train = pickle.load(open(os.path.join(data_dir, 'train.pkl'), 'rb'), encoding='bytes')
    train_x, train_y = train[b'x'], train[b'y']
    valid = pickle.load(open(os.path.join(data_dir, 'valid.pkl'), 'rb'), encoding='bytes')
    valid_x, valid_y = valid[b'x'], valid[b'y']
    test = pickle.load(open(os.path.join(data_dir, 'test.pkl'), 'rb'), encoding='bytes')
    test_x, test_y = test[b'x'], test[b'y']
    print((train_x).shape)
    return train_x, train_y, valid_x, valid_y, test_x, test_y


def load_imb_Page(data_dir):
    train = pickle.load(open(os.path.join(data_dir, 'train.pkl'), 'rb'), encoding='bytes')
    train_x, train_y = train[b'x'], train[b'y']
    valid = pickle.load(open(os.path.join(data_dir, 'valid.pkl'), 'rb'), encoding='bytes')
    valid_x, valid_y = valid[b'x'], valid[b'y']
    test = pickle.load(open(os.path.join(data_dir, 'test.pkl'), 'rb'), encoding='bytes')
    test_x, test_y = test[b'x'], test[b'y']
    print(train_x.shape)
    return train_x, train_y, valid_x, valid_y, test_x, test_y


def load_imb_Vehicle(data_dir):
    train = pickle.load(open(data_dir, 'rb'), encoding='bytes')
    train_x, train_y = train[b'x'], train[b'y']
    #valid = pickle.load(open('../data/real/spam/train.pkl', 'rb'), encoding='bytes')
    #valid_x, valid_y = valid[b'x'], valid[b'y']
    return train_x, train_y #valid_x, valid_y


def load_checker_board(data_dir):
    train = pickle.load(open(data_dir, 'rb'), encoding='bytes')
    train_x, train_y = train[b'x'], train[b'y']
    return train_x, train_y


def load_amazon(n_features, filename, source_domain, target_domain):
    """
    Load amazon reviews
    """

    def shuffle(x, y):
        """
        shuffle data (used by split)
        """
        index_shuf = np.arange(x.shape[0])
        np.random.shuffle(index_shuf)
        x = x[index_shuf, :]
        y = y[index_shuf]
        return x, y

    def to_one_hot(a):
        b = np.zeros((len(a), 2))
        b[np.arange(len(a)), a] = 1
        return b

    def split_data(d_s_ind, d_t_ind, x, y, offset, n_tr_samples, r_seed=0):
        # x = normalize(x, axis=0, norm='max')
        # x = np.log(1.+x)
        np.random.seed(r_seed)
        x_s_tr = x[offset[d_s_ind, 0]:offset[d_s_ind, 0] + n_tr_samples, :]
        x_t_tr = x[offset[d_t_ind, 0]:offset[d_t_ind, 0] + n_tr_samples, :]
        x_s_tst = x[offset[d_s_ind, 0] + n_tr_samples:offset[d_s_ind + 1, 0], :]
        x_t_tst = x[offset[d_t_ind, 0] + n_tr_samples:offset[d_t_ind + 1, 0], :]
        y_s_tr = y[offset[d_s_ind, 0]:offset[d_s_ind, 0] + n_tr_samples]
        y_t_tr = y[offset[d_t_ind, 0]:offset[d_t_ind, 0] + n_tr_samples]
        y_s_tst = y[offset[d_s_ind, 0] + n_tr_samples:offset[d_s_ind + 1, 0]]
        y_t_tst = y[offset[d_t_ind, 0] + n_tr_samples:offset[d_t_ind + 1, 0]]
        x_s_tr, y_s_tr = shuffle(x_s_tr, y_s_tr)
        x_t_tr, y_t_tr = shuffle(x_t_tr, y_t_tr)
        x_s_tst, y_s_tst = shuffle(x_s_tst, y_s_tst)
        x_t_tst, y_t_tst = shuffle(x_t_tst, y_t_tst)
        y_s_tr[y_s_tr == -1] = 0
        y_t_tr[y_t_tr == -1] = 0
        y_s_tst[y_s_tst == -1] = 0
        y_t_tst[y_t_tst == -1] = 0

        y_s_tr = to_one_hot(y_s_tr)
        y_t_tr = to_one_hot(y_t_tr)
        y_s_tst = to_one_hot(y_s_tst)
        y_t_tst = to_one_hot(y_t_tst)

        return x_s_tr, y_s_tr, x_t_tr, y_t_tr, x_s_tst, y_s_tst, x_t_tst, y_t_tst

    def turn_tfidf(x):
        df = (x > 0.).sum(axis=0)
        idf = np.log(1. * len(x) / (df + 1))
        return np.log(1. + x) * idf[None, :]

    def turn_one_hot(x):
        return (x > 0.).astype('float32')

    mat = loadmat(filename)

    xx=mat['xx']
    yy=mat['yy']
    offset=mat['offset']

    x=xx[:n_features,:].toarray().T #n_samples X n_features
    y=yy.ravel()

    x_s_tr, y_s_tr, x_t_tr, y_t_tr, x_s_tst, y_s_tst, x_t_tst, y_t_tst = split_data(source_domain,
                                                                                    target_domain,
                                                                                    x, y, offset, 2000)
    x = turn_tfidf(np.concatenate([x_s_tr, x_s_tst, x_t_tr, x_t_tst], axis=0))
    x_s = x[:len(x_s_tr) + len(x_s_tst)]
    x_t = x[len(x_s):]

    x_s_tr = np.copy(x_s[:len(x_s_tr)])
    x_s_tst = np.copy(x_s[len(x_s_tr):])

    x_t_tr = np.copy(x_t[:len(x_t_tr)])
    x_t_tst = np.copy(x_t[len(x_t_tr):])

    return x_s_tr, y_s_tr, x_t_tr, y_t_tr, x_t_tst, y_t_tst

