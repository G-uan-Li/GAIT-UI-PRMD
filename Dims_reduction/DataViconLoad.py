import csv
import numpy as np
import os

# print(np.__version__)
# exit() /root/autodl-tmp/data/
def load_data():
    f = open('/root/autodl-tmp/data/Data_Correct_m02.csv')
    # f = open('E:/gait/GAIT/Data/Data_correct.csv')
    csv_f = csv.reader(f)
    X_Corr = list(csv_f)

    # Convert the input sequences into numpy arrays
    train_input1 = np.asarray(X_Corr)
    n_dim = 117
    data_correct = np.zeros((90, 240, n_dim))
    for i in range(len(train_input1) // n_dim):
        data_correct[i, :, :] = np.transpose(train_input1[n_dim * i:n_dim * (i + 1), :])

    f = open('/root/autodl-tmp/data/Data_Incorrect_m02.csv')   # (90, 240, 117)
    # f = open('E:/gait/GAIT/Data/Data_Incorrect.csv')   #(90, 240, 117)
    csv_f = csv.reader(f)
    X_Incor = list(csv_f)

    # Convert the input sequences into numpy arrays
    train_input2 = np.asarray(X_Incor)
    n_dim = 117
    data_incorrect = np.zeros((90, 240, n_dim))
    for i in range(len(train_input2) // n_dim):
        data_incorrect[i, :, :] = np.transpose(train_input2[n_dim * i:n_dim * (i + 1), :])

    return data_correct, data_incorrect

data_correct, data_incorrect = load_data()

# Check the shape of loaded data
print("Shape of data_correct:", data_correct.shape)
print("Shape of data_incorrect:", data_incorrect.shape)
# Shape of data_correct: (90, 240, 117)
# Shape of data_incorrect: (90, 240, 117)