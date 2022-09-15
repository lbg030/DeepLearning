import numpy as np

import matplotlib.pyplot as plt

threshold = [0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]
scores = [(0.868, 83.21299638989169, 92.2), (0.869, 83.85321100917432, 91.4), (0.873, 85.38899430740038, 90.0), (0.87, 85.85271317829456, 88.6), (0.869, 86.2475442043222, 87.8), (0.876, 88.68312757201646, 86.2), (0.864, 89.56521739130436, 82.39999999999999), (0.852, 91.70616113744076, 77.4), (0.815, 94.61756373937678, 66.8)]
acc = []
recall = []
precision = []
f1_score = []
for score in scores:
    f1_score.append(score[1]*score[2]/(score[1]+score[2]) * 2)
    acc.append(score[0] * 100)
    precision.append(score[1])
    recall.append(score[2])
X = np.linspace(0.5, 0.9, 9)

Y1 = precision
Y2 = recall
Y3 = f1_score
Y4 = acc

plt.plot(X, Y1)

plt.plot(X, Y2)

plt.plot(X, Y3)

plt.plot(X, Y4)
plt.legend(('Precision','Recall','F1_Score','ACC'))
plt.show()