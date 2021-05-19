import torch
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import sys


class MultinomialLinearModel(nn.Module):
    # basic multinomial regression model 
    def __init__(self, in_features=4, n_categories=3):
        super(MultinomialLinearModel, self).__init__()
        assert n_categories >= 3
        self.w =  nn.Linear(in_features, n_categories)

    def forward(self, x):
        return F.softmax(self.w(x), dim=-1)


class MultinomialPerceptronModel(nn.Module):
    # basic perceptron multinomial regression model 
    def __init__(self, in_features=4, hidden_size=20, n_categories=3):
        super(MultinomialPerceptronModel, self).__init__()
        assert n_categories >= 3
        self.w1 = nn.Linear(in_features, hidden_size)
        self.w2 = nn.Linear(hidden_size, n_categories)
    def forward(self, x):
        return F.softmax(self.w2(torch.sigmoid(self.w1(x))), dim=-1)


class LogisticLinearModel(nn.Module):
    # basic logistic regression model 
    def __init__(self, in_features=4, n_categories=2):
        super(LogisticLinearModel, self).__init__()
        assert n_categories == 2
        self.w =  nn.Linear(in_features, 1)

    def forward(self, x):
        return torch.sigmoid(self.w(x))


class LogisticPerceptronModel(nn.Module):
    # basic logistic regression model 
    def __init__(self, in_features=4, hidden_size=20, n_categories=2):
        super(LogisticLinearModel, self).__init__()
        assert n_categories == 2
        self.w1 =  nn.Linear(in_features, hidden_size)
        self.w2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        return torch.sigmoid(self.w2(torch.sigmoid(self.w1(x))))


def convert_to_categorical(df, column=None):
    # one-hot encode labels for a pandas dataframe 
    # using categorical data in specified column 
    labels = df[column].unique()
    indices = np.arange(len(labels))
    mapping = {}

    identity = np.eye(len(labels))
    encoded = []

    for l, i in zip(labels, indices):
        mapping[l] = i

    for i in range(len(df)):
        encoded.append(identity[mapping[df.iloc[i][column]]])

    return np.stack(encoded), mapping


def dkl(pred, labels):
    # takes (n_samples x n_labels) preds and (n_samples x n_labels) ground truth
    # equivalent to minimizing KL assuming labels are fixed distribution
    # and take the mean across the batch
    # print(pred[labels.bool()].shape)
    return torch.mean(-1 * torch.log((pred[labels.bool()])))


def binary_cross_entropy(pred, labels):
    # labels should be of shape (n_samples x 2)
    # pred should be of shape (n_samples x 1)
    # one-hot encoded label
    # pred is the probability that a sample has label 1
    # 1-pred is the probability that a sample has label 0 
    batched_loss = labels * torch.log(1 / torch.squeeze(pred)) + \
    (1-labels) * torch.log(1 / (1 - torch.squeeze(pred)))
    return torch.mean(batched_loss)


def compute_accuracy(pred, labels):
    pred_np = pred.detach().numpy()
    labels_np = labels.detach().numpy()
   
    # non-binary classification setting  
    if pred_np.shape[1] > 1:    
        pred_idx = np.argmax(pred_np, axis=1)
        label_idx = np.argmax(labels_np, axis=1)
        return np.sum(pred_idx == label_idx) / len(pred_idx)

    # binary classification setting
    if pred_np.shape[1] == 1:
        return torch.sum(torch.squeeze(pred >= 0.5) ==  labels) / len(pred_np)


class ModelStorage:
    def __init__(self, trained_model, acc, mapping, col):
        self.model = trained_model
        self.acc = acc
        self.mapping = mapping
        self.col = col

    def __lt__(self, other):
        return self.acc < other.acc


def fit_categorical_lm(df, 
    cols, 
    embedding, 
    n_epochs=5000, 
    loss_fn=dkl,
    binary_loss_fn=binary_cross_entropy,
    model_t=MultinomialLinearModel,
    binary_model_t=LogisticLinearModel,
    lr=3e-4
    ):
    """Fit a series of Linear models that predict categorical variables using torch
    
    Assumes that each row of the embedding is aligned with each row of the dataframe (more in args description)

    Args:
        df: pandas dataframe of shape (n_samples x -1); each row is a patient + must contain all columns in cols
        cols: a list of strings that contain columns of categorical variables in df which we want to predict
        embedding: torch tensor of shape (n_samples x dim); each row is a patient, each column 
            is a feature that is computed via mpcca
        n_epochs: integer number of training steps; all models are trained using autodiff + Adam

        loss_fn: loss function (must be torch backpropable); default is KL divergence 
        binary_loss_fn: loss function for binary setting (must be torch backpropable); 
            default is binary cross-entropy

        model_t: class type for Linear Model (must be a pytorch nn.Module), default is Multinomial linear reg.        
        binary_model_t: class type for Linear Model for binary setting. 
            (must be a pytorch nn.Module), default is logistic reg.

    returns:
        all_models: list of ModelStorage objects that contain trained models, maps, and acc in order
            of columns 
    """
    all_models = []

    for col in cols:
        print('-'*100)
        print('Training on Column: {}'.format(col))
        # identify categorical labels + one-hot encode them
        labels, idx_map = convert_to_categorical(df, column=col)
        labels = torch.tensor(labels)
        n_cats = len(idx_map)

        # create model + set up optimizer
        embedding_dim = embedding.shape[1]
        cur_model_t = model_t if n_cats > 2 else binary_model_t
        cur_loss_fn = loss_fn if n_cats > 2 else binary_loss_fn
        labels = labels if n_cats > 2 else torch.argmax(labels, axis=1).double()

        regressor = cur_model_t(in_features=embedding_dim, n_categories=n_cats).double()
        optimizer = torch.optim.Adam(regressor.parameters(), lr=lr)
        optimizer.zero_grad()

        # train model 
        for epoch in range(n_epochs):
            y_hat = regressor(embedding)
            loss = cur_loss_fn(torch.squeeze(y_hat), labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if (epoch % 1000 == 0):
                print('epoch: {} acc: {} loss: {}'.format(epoch, compute_accuracy(y_hat, labels), loss.item()))

        all_models.append(ModelStorage(regressor, compute_accuracy(regressor(embedding), labels), idx_map, col))

    return all_models


def LinearRegression(mat, labels):
    # mat must be of shape (n_samples x k_features+1)
    # note: a column of ones is used to allow us to model 
    # affine functions as well
    mat = np.concatenate([mat, np.ones(len(mat), 1)], axis=1)
    lin_params = np.linalg.inv(mat.T @ mat) @ mat.T @ labels
    y_hat = mat @ lin_params
    return lin_params, y_hat

