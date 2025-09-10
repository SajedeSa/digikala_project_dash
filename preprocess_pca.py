import numpy as np
from sklearn.decomposition import PCA
import os

# مسیرهای ورودی و خروجی
DATA_DIR = "data"
X_SCALED_PATH = os.path.join(DATA_DIR, "X_scaled.npy")
X_PCA_PATH = os.path.join(DATA_DIR, "X_pca.npy")

print("Loading X_scaled.npy...")
# بررسی وجود فایل ورودی
if not os.path.exists(X_SCALED_PATH):
    print(f"Error: {X_SCALED_PATH} not found. Please make sure the file exists.")
else:
    X_scaled = np.load(X_SCALED_PATH)
    print(f"Data loaded with shape: {X_scaled.shape}")

    print("Performing PCA calculation... This may take a few minutes.")
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    print(f"PCA calculation complete. New shape: {X_pca.shape}")

    print(f"Saving the result to {X_PCA_PATH}...")
    np.save(X_PCA_PATH, X_pca)
    print("PCA data saved successfully. You can now run the main application.")
