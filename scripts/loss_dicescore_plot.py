import numpy as np
import matplotlib.pyplot as plt

# Initialize lists to store the values
loss_criterion = []
loss_dice = []
loss_total = []
dice_score = []

# Open and read the file
with open('loss_data.txt', 'r') as file:
    for line in file:
        if "Loss (criterion):" in line:
            loss_criterion.append(float(line.split(":")[1].strip()))
        elif "Loss (dice):" in line:
            loss_dice.append(float(line.split(":")[1].strip()))
        elif "Loss (total):" in line:
            loss_total.append(float(line.split(":")[1].strip()))
        elif "Dice score:" in line:
            dice_score.append(float(line.split(":")[1].strip()))



# Convert lists to numpy arrays
loss_criterion = np.array(loss_criterion)
loss_dice = np.array(loss_dice)
loss_total = np.array(loss_total)
dice_score = np.array(dice_score)
epoch = np.arange(0.2, 30 + 0.2, 0.2)

# Print the numpy arrays (optional)
# print("Losses (Criterion):", losses_criterion)
# print("Losses (Dice):", losses_dice)
# print("Losses (Total):", losses_total)
# print("Dice Scores:", dice_scores)

# Plot loss
plt.scatter(epoch, loss_criterion, label='Criterion', marker='x')
plt.scatter(epoch, loss_dice, label='Dice', marker='x')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training loss of model01')
plt.legend()
plt.show()

# Plot dice score
plt.scatter(epoch, dice_score, marker='x')
plt.xlabel('Epoch')
plt.ylabel('Dice score')
plt.title('Validation dice score of model01')
plt.show()
