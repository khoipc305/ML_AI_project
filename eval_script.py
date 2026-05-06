# evaluation script to collect model metrics for comparison, from eval_script import evaluate
# to use shared method evaluate
# sources: 
# http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
	f1_score, accuracy_score, roc_auc_score, confusion_matrix, 
	ConfusionMatrixDisplay, roc_curve, auc)

CLASSES = [0, 1, 2, 3, 4, 5, 6]

def evaluate(model_name: str, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray):
	"""
	model_name: str,
		model being evaluated
	y_true : 1D int array,
		actual class labels
	y_pred : 1D int array,
		predicted class labels
	y_prob : 2d float array (n_samples, n_classes),
		predicted probabilities for each class (rows must add to 1)
	"""


	macro_f1 = f1_score(y_true, y_pred, average='macro')
	acc = accuracy_score(y_true, y_pred)
	cm = confusion_matrix(y_true, y_pred)

	roc_aucs = {}
	for i, cls in enumerate(CLASSES):
		ovr_true = y_true == i	# isolate class to compare, one-vs-rest
		roc_aucs[cls] = roc_auc_score(ovr_true, y_prob[:, i])

	print(f"{model_name} Evaluation Results")
	print(f"Macro-Averaged F1 Score: {macro_f1:.4f}")
	print(f"Overall Accuracy: {acc:.4f}")
	print(f"ROC-AUC Per Class:")
	for cls, val in roc_aucs.items():
		print(f"{cls}: {val:.4f}")
	print(f"Confusion Matrix:\n{cm}\n")

	fig = plt.figure(figsize=(14, 5))
	fig.suptitle(f"{model_name} Evaluation Summary", fontsize=14, fontweight="bold")
	gs  = gridspec.GridSpec(1, 2, figure=fig)
	
	# confusion matrix
	ax_cm = fig.add_subplot(gs[0])
	disp  = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)
	disp.plot(ax=ax_cm, colorbar=False, cmap="Blues")
	ax_cm.set_title("Confusion Matrix")
	
	# ROC curves (one-vs-rest)
	ax_roc = fig.add_subplot(gs[1])
	for i, cls in enumerate(CLASSES):
		binary_true = y_true == i
		fpr, tpr, _ = roc_curve(binary_true, y_prob[:, i])
		roc_val = roc_aucs[cls]
		ax_roc.plot(fpr, tpr, label=f"{cls} (AUC={roc_val:.2f})")
	
	ax_roc.plot([0, 1], [0, 1], "k--", linewidth=0.8)
	ax_roc.set_xlabel("False Positive Rate")
	ax_roc.set_ylabel("True Positive Rate")
	ax_roc.set_title("ROC Curves (One-vs-Rest)")
	ax_roc.legend(loc="lower right")
	
	plt.tight_layout()
	fname = f"{model_name}_eval.png"
	plt.savefig(fname, dpi=120)
	plt.close()
	print(f"Figure saved as {fname}")