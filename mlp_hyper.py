import os 
import pandas as pd
import os
import numpy as np 
import random
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

# neural network method 
from sklearn.neural_network import MLPClassifier

# evaluation metrics
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    # for area in (0, 1, 2, 3, 4, 5, 6, 7):
    result = []
    for size in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
        area = 4
        print(f"==>Area_{area}: ")
        train_data, train_target = load_data_ml(area, 'train')
        valid_data, valid_target = load_data_ml(area, 'valid')
        print("Train Data Shape: ", train_data.shape, train_target.shape)
        print("Val Data Shape: ", valid_data.shape, valid_target.shape)

        clf = MLPClassifier(hidden_layer_sizes=(size,), max_iter=300, random_state=1)
        clf.fit(train_data, train_target)
        acc_train = round(clf.score(train_data, train_target) * 100, 3)
        print(f"Training Accuracy: {acc_train} %")
        result.append(acc_train)
    print(result)

    '''
        Fig. 2 in Project Report
    '''

    fig, ax = plt.subplots(figsize=(12, 6)) # 12， 4
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # plt.xticks(time, x_label)  # DAAGCN: FF7F0F AGCRN: 1F77B4; GT: 2AA02B
    # ax.set_ylim(0, 580)
    x = np.arange(10)
    x_label = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    plt.xticks(x, x_label)
    acc = result # [50.0, 50.0, 62.753, 56.883, 73.684, 74.089, 75.506, 86.235, 83.806, 91.7]
    # 2CA02B, D62728, 2777B4, 9467BD
    lns1 = ax.plot(x, acc, '-', color="#D62728", linewidth=2) 

    plt.xlabel("The number of neurons in the middle hidden layer", fontsize=15)
    plt.ylabel("Training accuracy (%)", fontsize=15)

    plt.grid(linestyle='--')
    # plt.legend(fontsize=10, loc='upper right') # loc='upper left'
    plt.savefig('mlp.png', format='png', bbox_inches='tight', pad_inches=0.05, dpi=1000)
    plt.show()