# knn.py
# K-Nearest Neighbours on HAM10000 skin lesion images.
# Trained on PCA-reduced features using Euclidean distance,
# with k selected using grid search over element k in set {3, 5, 7, 11}.


import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from config import TRAIN_LABELED_CSV, TEST_CSV
from utils import preprocess_image, normalize_image
from eval_script import evaluate


# image load helper
def load_images(csv):
	"""Load images from a given CSV file, return flat feature vectors and labels."""
	df = pd.read_csv(csv)

	X = []
	Y = []

	for _, row in df.iterrows():
		img = preprocess_image(row['image_path'])
		img = normalize_image(img)
		X.append(img.flatten())
		Y.append(row['label'])
	return np.array(X), np.array(Y)


# load training and testing data
x_train, y_train = load_images(TRAIN_LABELED_CSV)
x_test, y_test = load_images(TEST_CSV)

# apply PCA
pca = PCA(n_components=0.95)	# 95% variance retained
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)

# train KNN with grid search
given_grid = {'n_neighbors': [3, 5, 7, 11]}
grid_search = GridSearchCV(KNeighborsClassifier(metric='euclidean'), given_grid, scoring='f1_macro')	# f1 scoring to keep consistent comparisons in project
grid_search.fit(x_train, y_train)

# predict
best_knn = grid_search.best_estimator_
y_pred = best_knn.predict(x_test)
y_prob = best_knn.predict_proba(x_test)

# evaluate
evaluate("KNN", y_test, y_pred, y_prob)