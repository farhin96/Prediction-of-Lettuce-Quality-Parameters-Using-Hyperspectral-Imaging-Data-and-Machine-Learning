import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor


# =========================
# Load numeric spectral data
# =========================
data = pd.read_excel("GH_1and2_phase_bottom.xlsx", header=None)
data = data.apply(pd.to_numeric, errors="coerce").to_numpy()

# =========================
# Extract waveband and reflectance values
# =========================
waveband = data[0, 1:]          # MATLAB: data(1, 2:end)
reflectance = data[1:, 1:]      # MATLAB: data(2:end, 2:end)

# =========================
# Keep wavelengths up to 1009 nm
# =========================
valid_cols = waveband <= 1009
waveband = waveband[valid_cols]
reflectance = reflectance[:, valid_cols]

# =========================
# RFE settings
# =========================
numFinalFeatures = 30
stepSize = 5

# =========================
# Target rows and columns
# =========================
targetRows = list(range(2, 110))          # MATLAB 2:109
targetCols = [465, 466, 467, 468, 469]   # MATLAB 465:469
targetLabels = ["pH", "EC", "NO3", "Ca", "Brix"]

target_row_indices = np.array(targetRows) - 1


# =========================
# Loop through targets
# =========================
for targetCol, targetName in zip(targetCols, targetLabels):

    print("\n====================================================")
    print(f"Running RFE for {targetName} (targetCol = {targetCol})")
    print("====================================================")

    # MATLAB: y_all = data(targetRows, targetCol)
    y_all = data[target_row_indices, targetCol - 1]

    # Remove samples with NaN in X or y
    valid_rows = np.all(~np.isnan(reflectance), axis=1) & ~np.isnan(y_all)

    X = reflectance[valid_rows, :]
    y = y_all[valid_rows].ravel()

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

        selected_idx = np.delete(selected_idx, remove_local_idx)

    # =========================
    # Final selected wavelengths
    # =========================
    selected_wavelengths = waveband[selected_idx]
    selected_reflectance = reflectance[valid_rows, :][:, selected_idx]

    # Sort selected wavelengths
    sort_order = np.argsort(selected_wavelengths)
    selected_wavelengths = selected_wavelengths[sort_order]
    selected_idx = selected_idx[sort_order]
    selected_reflectance = selected_reflectance[:, sort_order]

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
    X_selected = selected_reflectance

    print(f"Original number of wavelengths: {len(waveband)}")
    print(f"Selected number of wavelengths for {targetName}: {len(selected_wavelengths)}")