import os

import numpy as np
# from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.svm import SVC

import gryds.grid_search as gs


RESULTS_PATH = '/home/thales/MEGA/backups/masters/thesis_results/ptbr/svm'
DATA_PATH = '/home/thales/MEGA/artificial-intelligence/databases/speech/reduced_lapsbm16k/reduced-lapsbm16k.csv'

data = np.loadtxt(DATA_PATH, delimiter=',', skiprows=1)
X, Y = data[:, :-2], data[:, -2]

model = SVC(kernel='poly')
optimizer = gs.GS(3, RESULTS_PATH)
optimizer.tune(model, X, Y,
               C=[0.001, 0.01, 0.1, 1.0, 10],
               gamma=[0.001, 0.01, 0.1, 1.0, 10],
               degree=[1, 2, 3, 4, 5])



