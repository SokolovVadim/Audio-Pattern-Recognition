import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Hardcoded confusion matrix (absolute values or percentages)
cm_percent = np.array([
    [54.07, 11.11],  # row for true class 0
    [7.41, 27.41]    # row for true class 1
])

# Plot
plt.figure(figsize=(4, 3))
sns.heatmap(cm_percent, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=["Predicted 0", "Predicted 1"],
            yticklabels=["True 0", "True 1"])
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.title("Confusion Matrix (%)")
plt.tight_layout()
plt.savefig("rf_cm_file_level_10000ms.png", dpi=600)
plt.show()
