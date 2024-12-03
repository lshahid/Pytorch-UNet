import numpy as np
import pdb
import pandas as pd
import matplotlib.pyplot as plt

# data_train = np.genfromtxt('data_train.csv', delimiter=',', skip_header=1)
# data_val = np.genfromtxt('data_val.csv', delimiter=',', skip_header=1)
data_train = pd.read_csv('data_train.csv')
data_val = pd.read_csv('data_val.csv')
data_train['Total Loss'] = data_train[' Criterion Loss'] + data_train[' Dice Loss']
data_val['Total Loss'] = data_val[' Criterion Loss'] + data_val[' Dice Loss']

n_epochs = 30
# n_train = 29193
# n_val = 16215

# Calculate metrics from training and validation
train_max = []
train_min = []
train_avg = []
val_max = []
val_min = []
val_avg = []

for epoch in range(1, n_epochs + 1):
    train_epoch_data = data_train.loc[data_train['Epoch'] == epoch]
    val_epoch_data = data_val.loc[data_val['Epoch'] == epoch]
    train_max_value = train_epoch_data['Total Loss'].max()
    train_min_value = train_epoch_data['Total Loss'].min()
    train_avg_value = train_epoch_data['Total Loss'].mean()
    val_max_value = val_epoch_data['Total Loss'].max()
    val_min_value = val_epoch_data['Total Loss'].min()
    val_avg_value = val_epoch_data['Total Loss'].mean()

    # pdb.set_trace()
    train_max.append(train_max_value)
    train_min.append(train_min_value)
    train_avg.append(train_avg_value)
    val_max.append(val_max_value)
    val_min.append(val_min_value)
    val_avg.append(val_avg_value)

# Plot loss curves
train_max_arr = np.array(train_max)
train_min_arr = np.array(train_min)
train_avg_arr = np.array(train_avg)
val_max_arr = np.array(val_max)
val_min_arr = np.array(val_min)
val_avg_arr = np.array(val_avg)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_max_arr, label='Max')
plt.plot(train_avg_arr, label='Avg')
plt.plot(train_min_arr, label='Min')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid()
plt.legend()
plt.title('Training')

plt.subplot(1, 2, 2)
plt.plot(val_max_arr, label='Max')
plt.plot(val_avg_arr, label='Avg')
plt.plot(val_min_arr, label='Min')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid()
plt.legend()
plt.title('Validation')

plt.suptitle('Model 005')
plt.show()

# pdb.set_trace()
