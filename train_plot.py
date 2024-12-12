import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def get_args():
    parser = argparse.ArgumentParser(description='Plot loss curves from training U-Net')
    parser.add_argument('--dir', metavar='DIR', help='Directory with training data', required=True)
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save loss curve plot')
    parser.add_argument('--viz', '-v', action='store_true', help='Visualize loss curve plot')

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    dir = args.dir
    data_train_dir = os.path.join(dir, 'data_train.csv')
    data_val_dir = os.path.join(dir, 'data_val.csv')

    data_train = pd.read_csv(data_train_dir)
    data_val = pd.read_csv(data_val_dir)
    # data_train['Total Loss'] = data_train[' Criterion Loss'] + data_train[' Dice Loss']
    # data_val['Total Loss'] = data_val[' Criterion Loss'] + data_val[' Dice Loss']

    # Extract metrics from training and validation
    n_epochs = int(max(data_train['Epoch']))
    model_number = dir.split('_')[-1]
    
    epoch_list = []

    train_loss_criterion_max_list = []
    train_loss_criterion_min_list = []
    train_loss_criterion_mean_list = []
    val_loss_criterion_max_list = []
    val_loss_criterion_min_list = []
    val_loss_criterion_mean_list = []

    train_loss_dice_max_list = []
    train_loss_dice_min_list = []
    train_loss_dice_mean_list = []
    val_loss_dice_max_list = []
    val_loss_dice_min_list = []
    val_loss_dice_mean_list = []

    train_loss_jaccard_max_list = []
    train_loss_jaccard_min_list = []
    train_loss_jaccard_mean_list = []
    val_loss_jaccard_max_list = []
    val_loss_jaccard_min_list = []
    val_loss_jaccard_mean_list = []

    train_loss_total_max_list = []
    train_loss_total_min_list = []
    train_loss_total_mean_list = []
    val_loss_total_max_list = []
    val_loss_total_min_list = []
    val_loss_total_mean_list = []

    for epoch in range(1, n_epochs + 1):
        train_epoch_data = data_train.loc[data_train['Epoch'] == epoch]
        val_epoch_data = data_val.loc[data_val['Epoch'] == epoch]

        train_loss_criterion_max_value = train_epoch_data['Criterion Loss'].max()
        train_loss_criterion_min_value = train_epoch_data['Criterion Loss'].min()
        train_loss_criterion_mean_value = train_epoch_data['Criterion Loss'].mean()
        val_loss_criterion_max_value = val_epoch_data['Criterion Loss'].max()
        val_loss_criterion_min_value = val_epoch_data['Criterion Loss'].min()
        val_loss_criterion_mean_value = val_epoch_data['Criterion Loss'].mean()

        train_loss_dice_max_value = train_epoch_data['Dice Loss'].max()
        train_loss_dice_min_value = train_epoch_data['Dice Loss'].min()
        train_loss_dice_mean_value = train_epoch_data['Dice Loss'].mean()
        val_loss_dice_max_value = val_epoch_data['Dice Loss'].max()
        val_loss_dice_min_value = val_epoch_data['Dice Loss'].min()
        val_loss_dice_mean_value = val_epoch_data['Dice Loss'].mean()

        train_loss_jaccard_max_value = train_epoch_data['Jaccard Loss'].max()
        train_loss_jaccard_min_value = train_epoch_data['Jaccard Loss'].min()
        train_loss_jaccard_mean_value = train_epoch_data['Jaccard Loss'].mean()
        val_loss_jaccard_max_value = val_epoch_data['Jaccard Loss'].max()
        val_loss_jaccard_min_value = val_epoch_data['Jaccard Loss'].min()
        val_loss_jaccard_mean_value = val_epoch_data['Jaccard Loss'].mean()

        train_loss_total_max_value = train_epoch_data['Total Loss'].max()
        train_loss_total_min_value = train_epoch_data['Total Loss'].min()
        train_loss_total_mean_value = train_epoch_data['Total Loss'].mean()
        val_loss_total_max_value = val_epoch_data['Total Loss'].max()
        val_loss_total_min_value = val_epoch_data['Total Loss'].min()
        val_loss_total_mean_value = val_epoch_data['Total Loss'].mean()

        epoch_list.append(epoch)

        train_loss_criterion_max_list.append(train_loss_criterion_max_value)
        train_loss_criterion_min_list.append(train_loss_criterion_min_value)
        train_loss_criterion_mean_list.append(train_loss_criterion_mean_value)
        val_loss_criterion_max_list.append(val_loss_criterion_max_value)
        val_loss_criterion_min_list.append(val_loss_criterion_min_value)
        val_loss_criterion_mean_list.append(val_loss_criterion_mean_value)

        train_loss_dice_max_list.append(train_loss_dice_max_value)
        train_loss_dice_min_list.append(train_loss_dice_min_value)
        train_loss_dice_mean_list.append(train_loss_dice_mean_value)
        val_loss_dice_max_list.append(val_loss_dice_max_value)
        val_loss_dice_min_list.append(val_loss_dice_min_value)
        val_loss_dice_mean_list.append(val_loss_dice_mean_value)

        train_loss_jaccard_max_list.append(train_loss_jaccard_max_value)
        train_loss_jaccard_min_list.append(train_loss_jaccard_min_value)
        train_loss_jaccard_mean_list.append(train_loss_jaccard_mean_value)
        val_loss_jaccard_max_list.append(val_loss_jaccard_max_value)
        val_loss_jaccard_min_list.append(val_loss_jaccard_min_value)
        val_loss_jaccard_mean_list.append(val_loss_jaccard_mean_value)

        train_loss_total_max_list.append(train_loss_total_max_value)
        train_loss_total_min_list.append(train_loss_total_min_value)
        train_loss_total_mean_list.append(train_loss_total_mean_value)
        val_loss_total_max_list.append(val_loss_total_max_value)
        val_loss_total_min_list.append(val_loss_total_min_value)
        val_loss_total_mean_list.append(val_loss_total_mean_value)

    # Plot loss curves
    epoch_arr = np.array(epoch_list)

    train_loss_criterion_max_arr = np.array(train_loss_criterion_max_list)
    train_loss_criterion_min_arr = np.array(train_loss_criterion_min_list)
    train_loss_criterion_mean_arr = np.array(train_loss_criterion_mean_list)
    val_loss_criterion_max_arr = np.array(val_loss_criterion_max_list)
    val_loss_criterion_min_arr = np.array(val_loss_criterion_min_list)
    val_loss_criterion_mean_arr = np.array(val_loss_criterion_mean_list)

    train_loss_dice_max_arr = np.array(train_loss_dice_max_list)
    train_loss_dice_min_arr = np.array(train_loss_dice_min_list)
    train_loss_dice_mean_arr = np.array(train_loss_dice_mean_list)
    val_loss_dice_max_arr = np.array(val_loss_dice_max_list)
    val_loss_dice_min_arr = np.array(val_loss_dice_min_list)
    val_loss_dice_mean_arr = np.array(val_loss_dice_mean_list)

    train_loss_jaccard_max_arr = np.array(train_loss_jaccard_max_list)
    train_loss_jaccard_min_arr = np.array(train_loss_jaccard_min_list)
    train_loss_jaccard_mean_arr = np.array(train_loss_jaccard_mean_list)
    val_loss_jaccard_max_arr = np.array(val_loss_jaccard_max_list)
    val_loss_jaccard_min_arr = np.array(val_loss_jaccard_min_list)
    val_loss_jaccard_mean_arr = np.array(val_loss_jaccard_mean_list)
    
    train_loss_total_max_arr = np.array(train_loss_total_max_list)
    train_loss_total_min_arr = np.array(train_loss_total_min_list)
    train_loss_total_mean_arr = np.array(train_loss_total_mean_list)
    val_loss_total_max_arr = np.array(val_loss_total_max_list)
    val_loss_total_min_arr = np.array(val_loss_total_min_list)
    val_loss_total_mean_arr = np.array(val_loss_total_mean_list)

    plt.figure(figsize=(12, 14))
    plt.subplot(4, 2, 1)
    plt.plot(epoch_arr, train_loss_criterion_max_arr, label='Max')
    plt.plot(epoch_arr, train_loss_criterion_mean_arr, label='Mean')
    plt.plot(epoch_arr, train_loss_criterion_min_arr, label='Min')
    plt.xlabel('Epoch')
    plt.ylabel('Criterion Loss')
    plt.grid()
    plt.legend()
    plt.title('Training')
    plt.ylim((0, 3))

    plt.subplot(4, 2, 2)
    plt.plot(epoch_arr, val_loss_criterion_max_arr, label='Max')
    plt.plot(epoch_arr, val_loss_criterion_mean_arr, label='Mean')
    plt.plot(epoch_arr, val_loss_criterion_min_arr, label='Min')
    plt.xlabel('Epoch')
    plt.ylabel('Criterion Loss')
    plt.grid()
    plt.legend()
    plt.title('Validation')
    plt.ylim((0, 3))

    plt.subplot(4, 2, 3)
    plt.plot(epoch_arr, train_loss_dice_max_arr, label='Max')
    plt.plot(epoch_arr, train_loss_dice_mean_arr, label='Mean')
    plt.plot(epoch_arr, train_loss_dice_min_arr, label='Min')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Loss')
    plt.grid()
    plt.legend()
    # plt.title('Training')
    plt.ylim((0, 3))

    plt.subplot(4, 2, 4)
    plt.plot(epoch_arr, val_loss_dice_max_arr, label='Max')
    plt.plot(epoch_arr, val_loss_dice_mean_arr, label='Mean')
    plt.plot(epoch_arr, val_loss_dice_min_arr, label='Min')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Loss')
    plt.grid()
    plt.legend()
    # plt.title('Validation')
    plt.ylim((0, 3))

    plt.subplot(4, 2, 5)
    plt.plot(epoch_arr, train_loss_jaccard_max_arr, label='Max')
    plt.plot(epoch_arr, train_loss_jaccard_mean_arr, label='Mean')
    plt.plot(epoch_arr, train_loss_jaccard_min_arr, label='Min')
    plt.xlabel('Epoch')
    plt.ylabel('Jaccard Loss')
    plt.grid()
    plt.legend()
    # plt.title('Training')
    plt.ylim((0, 3))

    plt.subplot(4, 2, 6)
    plt.plot(epoch_arr, val_loss_jaccard_max_arr, label='Max')
    plt.plot(epoch_arr, val_loss_jaccard_mean_arr, label='Mean')
    plt.plot(epoch_arr, val_loss_jaccard_min_arr, label='Min')
    plt.xlabel('Epoch')
    plt.ylabel('Jaccard Loss')
    plt.grid()
    plt.legend()
    # plt.title('Validation')
    plt.ylim((0, 3))

    plt.subplot(4, 2, 7)
    plt.plot(epoch_arr, train_loss_total_max_arr, label='Max')
    plt.plot(epoch_arr, train_loss_total_mean_arr, label='Mean')
    plt.plot(epoch_arr, train_loss_total_min_arr, label='Min')
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss')
    plt.grid()
    plt.legend()
    # plt.title('Training')
    plt.ylim((0, 3))

    plt.subplot(4, 2, 8)
    plt.plot(epoch_arr, val_loss_total_max_arr, label='Max')
    plt.plot(epoch_arr, val_loss_total_mean_arr, label='Mean')
    plt.plot(epoch_arr, val_loss_total_min_arr, label='Min')
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss')
    plt.grid()
    plt.legend()
    # plt.title('Validation')
    plt.ylim((0, 3))

    plt.suptitle('Model '+ model_number, fontsize=14)

    if args.viz:
        plt.show()

    if not args.no_save:
        plt.savefig(os.path.join(dir, ('model_' + model_number + '_train.png')))
