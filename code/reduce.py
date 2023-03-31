import torch
import numpy
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.decomposition import PCA
from sklearn.decomposition import DictionaryLearning
from sklearn.manifold import LocallyLinearEmbedding


def sfg_selection(x_train, x_test, y_train, num):
    # Create the SFG object and fit data on the training data
    sfg = SelectKBest(chi2, k=num)

    # Get the selected features.
    x_train_selected = sfg.fit_transform(x_train, y_train)
    x_test_selected = sfg.transform(x_test)

    return x_train_selected, x_test_selected


def rfe_selection(svm, x_train, x_test, y_train, num):
    # Create the RFE object and rank the features.
    rfe = RFE(estimator=svm, n_features_to_select=num, step=16)

    # Get the selected features.
    x_train_selected = rfe.fit_transform(x_train, y_train)
    x_test_selected = rfe.transform(x_test)

    return x_train_selected, x_test_selected


def feature_select(svm, x_train, x_test, y_train, num=128, mode='sfg'):
    if mode == 'sfg':
        x_train_selected, x_test_selected = sfg_selection(x_train, x_test, y_train, num)
    else:
        x_train_selected, x_test_selected = rfe_selection(svm, x_train, x_test, y_train, num)

    return x_train_selected, x_test_selected


def pca_project(x_train, x_test, num):
    # Create a PCA object with the specified number of components.
    pca = PCA(n_components=num)

    # Fit PCA on the training data and apply the transformation to both the training and testing sets.
    x_train_pca = pca.fit_transform(x_train)
    x_test_pca = pca.transform(x_test)

    return x_train_pca, x_test_pca


class Autoencoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Linear(input_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        x = torch.relu(self.encoder(x))
        x = self.decoder(x)
        return x


def encoder_project(x_train, x_test, y_train, hidden_size=256, n_epoch=100, lr=1e-4, bz=512):
    x_train_torch = torch.tensor(x_train, dtype=torch.float32)
    y_train_torch = torch.tensor(y_train, dtype=torch.long)
    train_dataset = TensorDataset(x_train_torch, y_train_torch)
    train_loader = DataLoader(train_dataset, batch_size=bz, shuffle=True)
    x_test_torch = torch.tensor(x_test, dtype=torch.float32)

    input_size = x_train.shape[1]
    model = Autoencoder(input_size, hidden_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in tqdm(range(n_epoch)):
        running_loss = 0.0
        min_loss = float('inf')
        count = 0
        patience = 10

        for data in train_loader:
            inputs, _ = data
            outputs = model(inputs)
            loss = criterion(outputs, inputs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        if avg_loss < min_loss:
            min_loss = avg_loss
            count = 0
        else:
            count += 1
            if count > patience:
                break

        if (epoch + 1) % 10 == 0:
            tqdm.write(f"Epoch [{epoch + 1}/{n_epoch}], Loss: {avg_loss:.4f}")

    x_train_encoded = torch.relu(model.encoder(x_train_torch)).detach().numpy()
    x_test_encoded = torch.relu(model.encoder(x_test_torch)).detach().numpy()

    return x_train_encoded, x_test_encoded


def feature_project(x_train, x_test, y_train, num=128, mode='pca'):
    if mode == 'pca':
        x_train_selected, x_test_selected = pca_project(x_train, x_test, num)
    else:
        x_train_selected, x_test_selected = \
            encoder_project(x_train, x_test, y_train, num)

    return x_train_selected, x_test_selected


def sc_learn(x_train, x_test, num):
    # Create a DictionaryLearning object with the specified number of atoms.
    dictionary_learning = DictionaryLearning(n_components=num, transform_algorithm='lasso_lars', random_state=233)

    # Fit the dictionary on the training data and obtain the sparse codes.
    x_train_sparse = dictionary_learning.fit_transform(x_train)
    x_test_sparse = dictionary_learning.transform(x_test)

    return x_train_sparse, x_test_sparse


def lle_learn(x_train, x_test, y_train, num):
    n_neighbors = 30

    # Create a LocallyLinearEmbedding object with the specified parameters.
    lle = LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=num, method='standard', random_state=233)

    # Fit LLE on the training data and apply the transformation to both the training and testing sets.
    x_train_lle = lle.fit_transform(x_train, y_train)
    x_test_lle = lle.transform(x_test)

    return x_train_lle, x_test_lle


def feature_learn(x_train, x_test, y_train, num=512, mode='lle'):
    if mode == 'sc':
        x_train_selected, x_test_selected = sc_learn(x_train, x_test, num)
    else:
        x_train_selected, x_test_selected = lle_learn(x_train, x_test, y_train, num)

    return x_train_selected, x_test_selected
