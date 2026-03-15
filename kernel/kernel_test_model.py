"""
Output based on the kernel method, This is mainly output and evaluation part.
Produces accuracy, ROC-AUC curves, and confusion matrices to analyze "Sitting/Standing" overlap and local density estimation performance.
"""
# Import packages
import numpy as np
import pandas as pd
import time
import os
import matplotlib.pyplot as plt
import seaborn as sns
import psutil
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize
from kernel_model import MixedWeightedKernelClassifier, load_har_mixed_dataset

# visiting data
ORIGINAL_ROOT = "/Users/augleovo/PycharmProjects/PythonProject spline/.venv/bin/UCI HAR Dataset"
data_root = "/Users/augleovo/PycharmProjects/PythonProject spline/.venv/bin/Integrated_HAR_Dataset_Sampled"


if __name__ == "__main__":
    # start calculate time
    overall_start = time.time()
    P_DIM = 50 
    
    # loading data
    X_tr, y_tr = load_har_mixed_dataset(split="train", p_cts=P_DIM)
    X_te, y_te = load_har_mixed_dataset(split="test", p_cts=P_DIM)
    
    # initialize the model
    model = MixedWeightedKernelClassifier(p_fun=9, p_cat=1, p_cts=P_DIM)

    # Running time of parameter
    print("\n[Running] Optimizing weights (omega) and bandwidth (h)...")
    param_select_start = time.time()
    model.fit(X_tr, y_tr) 
    param_select_time = time.time() - param_select_start

    # time of prediciton
    print("[Running] Computing kernel weights and predicting...")
    prediction_core_start = time.time()
    y_hat = model.predict(X_tr, y_tr, X_te)
    prediction_core_time = time.time() - prediction_core_start

    # After processing data
    y_hat_cleaned = np.nan_to_num(y_hat, nan=np.median(y_tr))
    y_pred = np.clip(np.round(y_hat_cleaned), 1, 6).astype(int)
    
    # Basic prediction
    total_acc = accuracy_score(y_te, y_pred)
    actions = ['Walking', 'Upstairs', 'Downstairs', 'Sitting', 'Standing', 'Laying']
    precision, recall, f1, _ = precision_recall_fscore_support(y_te, y_pred, average='macro')

    from sklearn.metrics import roc_auc_score

    # transfer One-vs-Rest to binomial to calculate AUC
    y_te_bin = label_binarize(y_te, classes=[1, 2, 3, 4, 5, 6])
    y_pred_bin = label_binarize(y_pred, classes=[1, 2, 3, 4, 5, 6])
    roc_auc_macro = roc_auc_score(y_te_bin, y_pred_bin, average='macro', multi_class='ovr')

    # Memory Usage
    process = psutil.Process(os.getpid())
    memory_usage_mb = process.memory_info().rss / (1024 * 1024)

    # end timing
    overall_time = time.time() - overall_start
    total_method_time = param_select_time + prediction_core_time

    # print report
    print("\n" + "="*60)
    print(f"{'Performance Metric':<40} {'Value'}")
    print("-" * 60)
    print(f"{'1. Parameter Optimization Time (LOOCV):':<40} {param_select_time:.4f}s")
    print(f"{'2. Kernel Prediction Core Time:':<40} {prediction_core_time:.4f}s")
    print(f"{'3. Precision (Macro):':<40} {precision:.4f}")
    print(f"{'4. Recall (Macro):':<40} {recall:.4f}")
    print(f"{'5. F1-Score (Macro):':<40} {f1:.4f}")
    print(f"{'5.5 ROC-AUC (Macro):':<40} {roc_auc_macro:.4f}")
    print(f"{'6. Final Test Accuracy:':<40} {total_acc*100:.2f}%")
    print(f"{'7. Memory Usage:':<40} {memory_usage_mb:.2f} MB")
    print(f"{'8. Overall System Runtime:':<40} {overall_time:.4f}s")
    print("="*60 + "\n")

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # Confusion Matrix
    cm = confusion_matrix(y_te, y_pred, labels=[1,2,3,4,5,6])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,0],
                xticklabels=actions, yticklabels=actions)
    axes[0,0].set_title(f"Confusion Matrix (Acc: {total_acc*100:.2f}%)")
    axes[0,0].set_ylabel('Actual Activity')
    axes[0,0].set_xlabel('Predicted Activity')

    # ROC curve
    y_te_bin = label_binarize(y_te, classes=[1,2,3,4,5,6])
    n_classes = y_te_bin.shape[1]
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_te_bin[:, i], (y_pred == (i+1)).astype(int))
        roc_auc = auc(fpr, tpr)
        axes[0,1].plot(fpr, tpr, label=f'{actions[i]} (AUC = {roc_auc:.2f})')
    axes[0,1].plot([0, 1], [0, 1], 'k--')
    axes[0,1].set_title("ROC Curves (One-vs-Rest)")
    axes[0,1].set_xlabel("False Positive Rate")
    axes[0,1].set_ylabel("True Positive Rate")
    axes[0,1].legend(loc="lower right")

    # Precision-Recall curve
    for i in range(n_classes):
        p, r, _ = precision_recall_curve(y_te_bin[:, i], (y_pred == (i+1)).astype(int))
        axes[1,0].plot(r, p, label=f'{actions[i]}')
    axes[1,0].set_title("Precision-Recall Curves")
    axes[1,0].set_xlabel("Recall")
    axes[1,0].set_ylabel("Precision")
    axes[1,0].legend(loc="lower left")

    # fitting plot, actual category vs continous estimation
    sns.regplot(x=y_te, y=y_hat_cleaned, ax=axes[1,1], 
                x_jitter=0.2, scatter_kws={'alpha':0.3, 'color':'gray', 's':10}, 
                line_kws={'color':'red'})
    axes[1,1].set_title("Signal Fitting: Ground Truth vs. Kernel Estimator")
    axes[1,1].set_xlabel("Actual Class Label")
    axes[1,1].set_ylabel("Continuous Prediction ($\hat{Y}$)")
    axes[1,1].set_xticks([1,2,3,4,5,6])

    plt.tight_layout()
    plt.show()

    # print classification report
    print("Classification Report:")
    print(classification_report(y_te, y_pred, target_names=actions))
