import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# =========================
# Load numeric spectral data
# =========================
data = pd.read_excel("completed_leave.xlsx", header=None)
data = data.apply(pd.to_numeric, errors="coerce").to_numpy()

# =========================
# Extract waveband and reflectance values
# =========================
waveband = data[:, 1]        # MATLAB column 2
reflectance = data[:, 2:]    # MATLAB columns 3:end

# =========================
# Keep wavelengths up to 1009 nm
# =========================
valid_rows = waveband <= 1009
waveband = waveband[valid_rows]
reflectance = reflectance[valid_rows, :]

# =========================
# Arrange data for PCA
# rows = samples, columns = wavelengths
# =========================
X = reflectance.T

# =========================
# Standardize data
# =========================
scaler = StandardScaler()
Xz = scaler.fit_transform(X)

# =========================
# Run PCA
# =========================
pca = PCA()
score = pca.fit_transform(Xz)
coeff = pca.components_.T
latent = pca.explained_variance_
explained = pca.explained_variance_ratio_ * 100

# =========================
# Plot cumulative explained variance
# =========================
plt.figure()
plt.plot(np.cumsum(explained), marker="o", linewidth=1.5)
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance (%)")
plt.title("PCA Explained Variance")
plt.grid(True)
plt.tight_layout()
plt.show()

# =========================
# Choose number of PCs
# =========================
numPCs = np.argmax(np.cumsum(explained) >= 95) + 1
print(f"Number of PCs selected: {numPCs}")

# =========================
# PCA-based wavelength importance
# =========================
loading_importance = np.sum(np.abs(coeff[:, :numPCs]), axis=1)

# =========================
# Define spectral regions
# =========================
region1 = (waveband >= 350) & (waveband < 500)
region2 = (waveband >= 500) & (waveband < 700)
region3 = (waveband >= 700) & (waveband <= 1009)

# =========================
# Number of wavelengths to select from each region
# =========================
n1 = 10
n2 = 10
n3 = 10

# =========================
# Select top wavelengths from each region
# =========================
idx1_all = np.where(region1)[0]
idx2_all = np.where(region2)[0]
idx3_all = np.where(region3)[0]


def top_k_indices(values, k):
    k = min(k, len(values))
    if k == 0:
        return np.array([], dtype=int)
    return np.argsort(values)[-k:][::-1]


loc1 = top_k_indices(loading_importance[idx1_all], n1)
loc2 = top_k_indices(loading_importance[idx2_all], n2)
loc3 = top_k_indices(loading_importance[idx3_all], n3)

idx1 = idx1_all[loc1]
idx2 = idx2_all[loc2]
idx3 = idx3_all[loc3]

# =========================
# Combine selected indices
# =========================
idx_selected = np.concatenate([idx1, idx2, idx3])

# =========================
# Sort selected wavelengths
# =========================
sort_idx = np.argsort(waveband[idx_selected])
idx_selected = idx_selected[sort_idx]

selected_wavelengths = waveband[idx_selected]
selected_reflectance = reflectance[idx_selected, :]

# =========================
# Display selected wavelengths
# =========================
print("Selected wavelengths using forced multi-region PCA selection:")
print(selected_wavelengths)

# =========================
# Plot wavelength importance with selected bands
# =========================
plt.figure()
plt.plot(waveband, loading_importance, linewidth=1.5)
plt.scatter(
    selected_wavelengths,
    loading_importance[idx_selected],
    s=60
)
plt.xlabel("Wavelength (nm)")
plt.ylabel("PCA Loading Importance")
plt.title("Forced Multi-Region PCA Feature Selection")
plt.grid(True)
plt.tight_layout()
plt.show()

# =========================
# Reduced predictor matrix for modeling
# =========================
X_selected = selected_reflectance.T   # samples x selected wavelengths

print(f"Original number of wavelengths: {len(waveband)}")
print(f"Selected number of wavelengths: {len(selected_wavelengths)}")