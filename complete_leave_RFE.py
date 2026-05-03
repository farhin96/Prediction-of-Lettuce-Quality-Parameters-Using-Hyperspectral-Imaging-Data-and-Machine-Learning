import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor


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
# RFE settings
# =========================
numFinalFeatures = 30
stepSize = 5

# MATLAB rows 468:472 converted to Python indices 467:471
targetRows = [ 469, 470, 471, 472, 473]
targetLabels = ["pH", "EC", "NO3", "Ca", "Brix"]

# =========================
# Loop through targets
# =========================
for targetRow, targetName in zip(targetRows, targetLabels):

    print("\n====================================================")
    print(f"Running RFE for {targetName} (targetRow = {targetRow + 1})")
    print("====================================================")

    y_all = data[targetRow, 2:]

    # Remove samples with NaN in X or y
    valid_cols = np.all(~np.isnan(reflectance), axis=0) & ~np.isnan(y_all)

    reflectance_use = reflectance[:, valid_cols]
    y = y_all[valid_cols].ravel()

    # Arrange predictors: rows = samples, cols = wavelengths
    X = reflectance_use.T

    # Standardize predictors
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Initialize RFE
    selected_idx = np.arange(X.shape[1])
    history = []

    print(f"Starting with {len(selected_idx)} wavelengths...")

    while len(selected_idx) > numFinalFeatures:

        X_curr = X[:, selected_idx]

        # Random Forest regressor
        rf = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )

        rf.fit(X_curr, y)

        # Predictor importance
        imp = rf.feature_importances_

        # Sort by importance, ascending
        sortIdx = np.argsort(imp)

        # Remove least important features
        numRemove = min(stepSize, len(selected_idx) - numFinalFeatures)
        remove_local_idx = sortIdx[:numRemove]

        # Save progress
        history.append([
            len(selected_idx),
            np.mean(imp),
            np.min(imp),
            np.max(imp)
        ])

        print(
            f"RFE step: {len(selected_idx)} -> removing {numRemove} wavelengths, "
            f"{len(selected_idx) - numRemove} remain"
        )

        # Delete selected local indices
        selected_idx = np.delete(selected_idx, remove_local_idx)

    # =========================
    # Final selected wavelengths
    # =========================
    selected_wavelengths = waveband[selected_idx]
    selected_reflectance = reflectance_use[selected_idx, :]

    # Sort selected wavelengths
    sort_order = np.argsort(selected_wavelengths)
    selected_wavelengths = selected_wavelengths[sort_order]
    selected_idx = selected_idx[sort_order]
    selected_reflectance = selected_reflectance[sort_order, :]

    # Display selected wavelengths
    print(f"\nSelected wavelengths for {targetName}:")
    print(selected_wavelengths)

    # =========================
    # Plot selected wavelengths
    # =========================
    selected_mask = np.isin(np.arange(len(waveband)), selected_idx)

    plt.figure()
    plt.stem(waveband, selected_mask.astype(int))
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Selected (1 = yes)")
    plt.title(f"RFE Selected Wavelengths for {targetName}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # =========================
    # Final importance plot
    # =========================
    X_final = X[:, selected_idx]

    rf_final = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    rf_final.fit(X_final, y)

    final_imp = rf_final.feature_importances_

    plt.figure()
    plt.bar(selected_wavelengths, final_imp)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Importance")
    plt.title(f"Final RFE Selected Wavelength Importance for {targetName}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # =========================
    # Reduced predictor matrix for later modeling
    # =========================
    X_selected = selected_reflectance.T

    print(f"Original number of wavelengths: {len(waveband)}")
    print(f"Selected number of wavelengths for {targetName}: {len(selected_wavelengths)}")