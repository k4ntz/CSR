from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn import kernel_ridge
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import torch
import numpy as np
import scipy as sp
from typing import Union
from typing_extensions import Literal
import ipdb as pdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

__Mode = Literal["r2"]

def _disentanglement(z, hz, mode: __Mode = "r2", reorder=None):
    """Measure how well hz reconstructs z measured either by the Coefficient of Determination or the
    Pearson/Spearman correlation coefficient."""

    assert mode in ("r2", "accuracy")

    if mode == "r2":
        return metrics.r2_score(z, hz), None
    elif mode == "accuracy":
        return metrics.accuracy_score(z, hz), None

def nonlinear_disentanglement(z, hz, mode: __Mode = "r2", alpha=1.0, gamma=None, train_mode=False, model=None, scaler_z=None, scaler_hz=None):
    """Calculate disentanglement up to nonlinear transformations.

    Args:
        z: Ground-truth latents.
        hz: Reconstructed latents.
        mode: Can be r2, pearson, spearman
        train_test_split: Use first half to train linear model, second half to test.
            Is only relevant if there are less samples then latent dimensions.
    """
    if torch.is_tensor(hz):
        hz = hz.detach().cpu().numpy()
    if torch.is_tensor(z):
        z = z.detach().cpu().numpy()

    assert isinstance(z, np.ndarray), "Either pass a torch tensor or numpy array as z"
    assert isinstance(hz, np.ndarray), "Either pass a torch tensor or numpy array as hz"

    # # split z, hz to get train and test set for linear model
    # if train_test_split:
    #     n_train = len(z) // 2
    #     z_1 = z[:n_train]
    #     hz_1 = hz[:n_train]
    #     z_2 = z[n_train:]
    #     hz_2 = hz[n_train:]
    #     model = kernel_ridge.KernelRidge(kernel='linear', alpha=alpha, gamma=gamma)
    #     model.fit(hz_1, z_1)
    #     hz_2 = model.predict(hz_2)

    #     inner_result = _disentanglement(z_2, hz_2, mode=mode, reorder=False)

    #     return inner_result, (z_2, hz_2)
    # else:
    if train_mode:
        model = GridSearchCV(
            kernel_ridge.KernelRidge(kernel='rbf', gamma=0.1),
            param_grid={"alpha": [1e0, 0.1, 1e-2, 1e-3], "gamma": np.logspace(-2, 2, 4)}, 
            cv=3, 
            n_jobs=-1
        )
        model.fit(hz, z)
        return model
    else:
        hz = model.predict(hz)
        inner_result = _disentanglement(z, hz, mode=mode, reorder=False)
        return inner_result, (z, hz)

def compute_r2(train_z, train_hz, test_z, test_hz, require_tsne=False, save_path=None):
    # predict train_z using hz
    # normalize the first half and keep the normalization constants
    scaler_hz = StandardScaler()
    train_hz = scaler_hz.fit_transform(train_hz)
    scaler_z = StandardScaler()
    train_z = scaler_z.fit_transform(train_z)
    
    if require_tsne and train_z.shape[-1] > 1:
        tsne_z = TSNE(n_components=2)
        tsne_hz = TSNE(n_components=2)
        tsne_z_data = tsne_z.fit_transform(train_z)
        tsne_hz_data = tsne_hz.fit_transform(train_hz)
        # tsne_z_data = np.vstack((tsne_z_data.T, np.array([['s'] * train_z.shape[0]]))).T 
        # tsne_hz_data = np.vstack((tsne_hz_data.T, np.array([['s hat'] * train_z.shape[0]]))).T 
        # all_data = np.vstack((tsne_z_data, tsne_hz_data))
        df_z = pd.DataFrame(tsne_z_data, columns=['Dim1', 'Dim2'])
        df_z['Class'] =  np.array(['s'] * train_z.shape[0])
        
        df_hz = pd.DataFrame(tsne_hz_data, columns=['Dim1', 'Dim2'])
        df_hz['Class'] =  np.array(['s hat'] * train_hz.shape[0])
        all_data = pd.concat([df_z, df_hz], ignore_index=True)
        pre = save_path.split('.')[0]
        plt.figure(figsize=(8, 8)) 
        sns.scatterplot(data=all_data, hue='Class', x='Dim1', y='Dim2') 
        plt.savefig(pre + '_s.png')
        # plt.figure(figsize=(8, 8)) 
        # sns.scatterplot(data=df_hz, hue='Class', x='Dim1', y='Dim2') 
        # plt.savefig(pre + '_s_hat.png')

    # normalize the second one with the normalization constants
    test_hz = scaler_hz.transform(test_hz)
    test_z = scaler_z.transform(test_z)

    # train the model with the normalized first half
    model = nonlinear_disentanglement(train_z, train_hz, train_mode=True)

    # apply the model to infer on the second normalized half.
    r2, _ = nonlinear_disentanglement(test_z, test_hz, model=model, train_mode=False)

    return r2[0]

def test_independence(train_data_dict, test_data_dict, data_size_dict):
    train_s1 = train_data_dict['hs1']
    train_s2 = train_data_dict['hs2']
    train_s3 = train_data_dict['hs3']
    train_s4 = train_data_dict['hs4']
    train_action = train_data_dict['action'][:-1, ...]
    
    test_s1 = test_data_dict['hs1']
    test_s2 = test_data_dict['hs2']
    test_s3 = test_data_dict['hs3']
    test_s4 = test_data_dict['hs4']
    test_action = test_data_dict['action'][:-1, ...]
    
    train_data = np.concatenate((train_s1, train_s2, train_s3, train_s4), axis=-1)
    test_data = np.concatenate((test_s1, test_s2, test_s3, test_s4), axis=-1)
    scaler_data = StandardScaler()
    train_data = scaler_data.fit_transform(train_data)
    test_data = scaler_data.transform(test_data)
    
    train_x = train_data[:-1, ...]
    train_y = train_data[1:, ...]
    test_x = test_data[:-1, ...]
    test_y = test_data[1:, ...]
    
    train_x = np.concatenate((train_x, train_action), axis=-1)
    test_x = np.concatenate((test_x, test_action), axis=-1)
    
    model = GridSearchCV(
            kernel_ridge.KernelRidge(kernel='rbf', gamma=0.1),
            param_grid={"alpha": [1e0, 0.1, 1e-2, 1e-3], "gamma": np.logspace(-2, 2, 4)}, 
            cv=3, 
            n_jobs=-1
        )
    model.fit(train_x, train_y)
    predicted_test_y = model.predict(test_x)
    residual_test_y = predicted_test_y - test_y
    d1, d2, d3, d4 = data_size_dict['s1'], data_size_dict['s1'] + data_size_dict['s2'], data_size_dict['s1'] + data_size_dict['s2'] + data_size_dict['s3'], data_size_dict['s1'] + data_size_dict['s2'] + data_size_dict['s3'] + data_size_dict['s4']
    residual_s1, residual_s2, residual_s3, residual_s4, _ = np.split(residual_test_y, [d1, d2, d3, d4], axis=-1)
    [residual_s1_train, residual_s1_test] = np.split(residual_s1, 2, axis=0)
    [residual_s2_train, residual_s2_test] = np.split(residual_s2, 2, axis=0)
    [residual_s3_train, residual_s3_test] = np.split(residual_s3, 2, axis=0)
    [residual_s4_train, residual_s4_test] = np.split(residual_s4, 2, axis=0)
    
    
    s12 = compute_r2(residual_s2_train, residual_s1_train, residual_s2_test, residual_s1_test)
    s13 = compute_r2(residual_s3_train, residual_s1_train, residual_s3_test, residual_s1_test)
    s14 = compute_r2(residual_s4_train, residual_s1_train, residual_s4_test, residual_s1_test)
    
    s21 = compute_r2(residual_s1_train, residual_s2_train, residual_s1_test, residual_s2_test)
    s23 = compute_r2(residual_s3_train, residual_s2_train, residual_s3_test, residual_s2_test)
    s24 = compute_r2(residual_s4_train, residual_s2_train, residual_s4_test, residual_s2_test)
    
    s31 = compute_r2(residual_s1_train, residual_s3_train, residual_s1_test, residual_s3_test)
    s32 = compute_r2(residual_s2_train, residual_s3_train, residual_s2_test, residual_s3_test)
    s34 = compute_r2(residual_s4_train, residual_s3_train, residual_s4_test, residual_s3_test)
    
    s41 = compute_r2(residual_s1_train, residual_s4_train, residual_s1_test, residual_s4_test)
    s42 = compute_r2(residual_s2_train, residual_s4_train, residual_s2_test, residual_s4_test)
    s43 = compute_r2(residual_s3_train, residual_s4_train, residual_s3_test, residual_s4_test)
    
    return s12, s13, s14, s21, s23, s24, s31, s32, s34, s41, s42, s43