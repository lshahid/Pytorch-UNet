import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_train = pd.read_csv('data_train.csv')
data_val = pd.read_csv('data_val.csv')
data_train['Total Loss'] = data_train[' Criterion Loss'] + data_train[' Dice Loss']
data_val['Total Loss'] = data_val[' Criterion Loss'] + data_val[' Dice Loss']

n_epochs = 30

# Calculate metrics from training and validation
epoch_list = []
train_max_list = []
train_min_list = []
train_avg_list = []
val_max_list = []
val_min_list = []
val_avg_list = []

for epoch in range(1, n_epochs + 1):
    train_epoch_data = data_train.loc[data_train['Epoch'] == epoch]
    val_epoch_data = data_val.loc[data_val['Epoch'] == epoch]
    train_max_value = train_epoch_data['Total Loss'].max()
    train_min_value = train_epoch_data['Total Loss'].min()
    train_avg_value = train_epoch_data['Total Loss'].mean()
    val_max_value = val_epoch_data['Total Loss'].max()
    val_min_value = val_epoch_data['Total Loss'].min()
    val_avg_value = val_epoch_data['Total Loss'].mean()

    epoch_list.append(epoch)
    train_max_list.append(train_max_value)
    train_min_list.append(train_min_value)
    train_avg_list.append(train_avg_value)
    val_max_list.append(val_max_value)
    val_min_list.append(val_min_value)
    val_avg_list.append(val_avg_value)

# Plot loss curves
epoch_arr = np.array(epoch_list)
train_max_arr = np.array(train_max_list)
train_min_arr = np.array(train_min_list)
train_avg_arr = np.array(train_avg_list)
val_max_arr = np.array(val_max_list)
val_min_arr = np.array(val_min_list)
val_avg_arr = np.array(val_avg_list)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epoch_arr, train_max_arr, label='Max')
plt.plot(epoch_arr, train_avg_arr, label='Avg')
plt.plot(epoch_arr, train_min_arr, label='Min')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid()
plt.legend()
plt.title('Training')
plt.ylim((0, 2.5))

plt.subplot(1, 2, 2)
plt.plot(epoch_arr, val_max_arr, label='Max')
plt.plot(epoch_arr, val_avg_arr, label='Avg')
plt.plot(epoch_arr, val_min_arr, label='Min')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid()
plt.legend()
plt.title('Validation')
plt.ylim((0, 2.5))

plt.suptitle('Model 007')
# plt.show()
plt.savefig('model_007_train.png')
