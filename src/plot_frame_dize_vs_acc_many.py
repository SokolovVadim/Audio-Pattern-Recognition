import matplotlib.pyplot as plt

frame_sizes = [20, 30, 100, 250, 500, 1000, 5000, 10000, 15000, 20000, 30000]

# accuracies_log = [0.6455, 0.6506, 0.6635, 0.6583, 0.6640, 0.6684, 0.7067, 0.7255, 0.7290, 0.7317, 0.7074]
# f1_scores_log = [0.6278, 0.6296, 0.6337, 0.6269, 0.6300, 0.6296, 0.5829, 0.6400, 0.6849, 0.6786, 0.7074]

# accuracies_rf = [0.6799, 0.6829, 0.6788, 0.6661, 0.6618, 0.6589, 0.6693, 0.6894, 0.6898, 0.6477, 0.6352]
# f1_scores_rf = [0.6115, 0.6115, 0.6027, 0.5907, 0.5945, 0.5957, 0.5799, 0.6006, 0.6398, 0.5925, 0.5873]

# accuracies_svm = [0.6635, 0.6736, 0.6824, 0.6996, 0.6890, 0.6916, 0.7186, 0.7576, 0.7245, 0.7575, 0.7556]
# f1_scores_svm = [0.5833, 0.5908, 0.5976, 0.6053, 0.5854, 0.5823, 0.5217, 0.6196, 0.6455, 0.6427, 0.7054]

accuracies_log = [0.7415, 0.7542, 0.7669, 0.7627, 0.7350, 0.7301, 0.7791, 0.7630, 0.7071, 0.6463, 0.7167]
f1_scores_log = [0.6433, 0.6667, 0.6746, 0.6627, 0.6310, 0.6164, 0.6885, 0.6444, 0.6420, 0.5672, 0.6909]

accuracies_rf = [0.7966, 0.7966, 0.7839, 0.7585, 0.7393, 0.7168, 0.8023, 0.8148, 0.7576, 0.7317, 0.8000]
f1_scores_rf = [0.6923, 0.6883, 0.6752, 0.6460, 0.6303, 0.6279, 0.7302, 0.7475, 0.7000, 0.7027, 0.7931]

accuracies_svm = [0.7873, 0.7992, 0.8263, 0.8263, 0.7949, 0.7876, 0.8140, 0.8000, 0.7273, 0.7317, 0.7333]
f1_scores_svm = [0.6694, 0.6749, 0.6963, 0.6822, 0.6190, 0.6000, 0.6735, 0.6897, 0.6494, 0.6071, 0.7037]


plt.figure(figsize=(10, 6))

# Logistic Regression
plt.plot(frame_sizes, accuracies_log, marker='o', label="LogReg Accuracy", linestyle='-')
plt.plot(frame_sizes, f1_scores_log, marker='x', label="LogReg F1", linestyle='--')

# Random Forest
plt.plot(frame_sizes, accuracies_rf, marker='o', label="RF Accuracy", linestyle='-')
plt.plot(frame_sizes, f1_scores_rf, marker='x', label="RF F1", linestyle='--')

# SVM
plt.plot(frame_sizes, accuracies_svm, marker='o', label="SVM Accuracy", linestyle='-')
plt.plot(frame_sizes, f1_scores_svm, marker='x', label="SVM F1", linestyle='--')

plt.xlabel("Frame Size (ms)")
plt.ylabel("Score")
plt.title("Model Performance with added features vs Frame Size (Log Scale)")
plt.xscale("log")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.show()
