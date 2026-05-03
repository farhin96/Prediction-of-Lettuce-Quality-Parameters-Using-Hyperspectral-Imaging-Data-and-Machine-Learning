import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# =========================
# Load numeric spectral data
# =========================
data = pd.read_excel("GH_1and2_phase_bottom.xlsx", header=None)
data = data.apply(pd.to_numeric, errors="coerce").to_numpy()

# =========================
# Extract waveband and reflectance values
# =========================
# MATLAB: data(1, 2:end)
waveband = data[0, 1:]

# MATLAB: data(2:end, 2:end)
reflectance = data[1:, 1:]

# =========================
# Keep wavelengths up to 1009 nm
# =========================
valid_cols = waveband <= 1009
waveband = waveband[valid_cols]
reflectance = reflectance[:, valid_cols]

# =========================
# Arrange data for PCA
# rows = samples, columns = wavelengths
# =========================
X = reflectance

# =========================
# Remove rows with NaN values
# =========================
valid_samples = np.all(~np.isnan(X), axis=1)
X = X[valid_samples, :]

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


def top_k_from_region(indices, importance, k):
    if len(indices) == 0:
        return np.array([], dtype=int)

    k = min(k, len(indices))
    region_importance = importance[indices]

    local_order = np.argsort(region_importance)[-k:][::-1]
    return indices[local_order]


idx1 = top_k_from_region(idx1_all, loading_importance, n1)
idx2 = top_k_from_region(idx2_all, loading_importance, n2)
idx3 = top_k_from_region(idx3_all, loading_importance, n3)

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
selected_reflectance = reflectance[:, idx_selected]

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
X_selected = selected_reflectance  # samples x selected wavelengths

print(f"Original number of wavelengths: {len(waveband)}")
print(f"Selected number of wavelengths: {len(selected_wavelengths)}")