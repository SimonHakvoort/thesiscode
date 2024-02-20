import matplotlib.pyplot as plt
import pickle as pkl

import tensorflow as tf
from src.models.emos import EMOS
import numpy as np

from src.models.get_data import get_tensors, sort_tensor


path = '/net/pc200239/nobackup/users/hakvoort/models/emos_loss_CRPS_sample_distr_trunc_normal.pkl'

with open(path, 'rb') as f:
    emos_dict = pkl.load(f)

emos_model = EMOS(emos_dict)

x = np.linspace(0, 20, 100)

X_val, y_val, variances_val = get_tensors(emos_dict['neighbourhood_size'], emos_dict['features'], 0)

X_val = (X_val - emos_model.feature_mean) / emos_model.feature_std

X_val, y_val, variances_val = sort_tensor(X_val, y_val, variances_val)

loss = []
for i in x:
    loss_i = emos_model.Brier_score(X_val, y_val, variances_val, i)
    loss.append(loss_i.numpy())

plt.plot(x, loss)
plt.show()

