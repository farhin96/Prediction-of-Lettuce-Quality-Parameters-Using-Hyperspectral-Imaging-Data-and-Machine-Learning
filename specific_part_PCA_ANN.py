import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import random

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr


# =========================
# User settings
# =========================
specFile = "GH_1and2_phase_bottom.xlsx"

# MATLAB rows/columns
targetRows = list(range(2, 110))          # MATLAB 2:109
targetCols = [465, 466, 467, 468, 469]   # MATLAB 465:469
targetLabels = ["pH", "EC", "NO3", "Ca", "Brix"]

numRuns = 150
numTopRuns = 100

# Improved ANN settings
hidden_layer_sizes = (10,)
maxEpochs = 1000
regularization = 0.1
activation = "tanh"
solver = "lbfgs"

# 70 / 15 / 15 split
trainRatio = 0.70
testRatio = 0.15
validRatio = 0.15

# Selected wavelengths
wavebands_of_interest = np.array([
    486.93, 488.25, 489.58, 490.90, 492.22,
    493.55, 494.87, 496.20, 497.52, 498.85,
    680.00, 681.34, 682.68, 684.01, 685.35,
    686.69, 688.03, 689.37, 690.71, 692.05,
    730.96, 732.31, 733.65, 734.99, 736.34,
    737.68, 739.03, 740.37, 741.71, 743.06
])


# =========================
# Metric helpers
# =========================
def nrmse(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    denom = np.max(y_true) - np.min(y_true) + np.finfo(float).eps

    return rmse / denom


def corr_safe(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    if len(y_true) != len(y_pred):
        raise ValueError("y and y_pred must have the same length.")

    if np.any(~np.isfinite(y_true)) or np.any(~np.isfinite(y_pred)):
        return 0.0

    if np.var(y_true) == 0 or np.var(y_pred) == 0:
        return 0.0

    r, _ = pearsonr(y_true, y_pred)

    if np.isnan(r):
        return 0.0

    return r


# =========================
# ANN training function
# =========================
def train_ann_one(Xtr, Ytr, Xva, Yva, Xte, seed):
    # Standardize X using training set only
    scaler_X = StandardScaler()
    XtrZ = scaler_X.fit_transform(Xtr)
    XvaZ = scaler_X.transform(Xva)
    XteZ = scaler_X.transform(Xte)

    # Standardize Y using training set only
    scaler_Y = StandardScaler()
    YtrZ = scaler_Y.fit_transform(Ytr.reshape(-1, 1)).ravel()

    model = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver=solver,
        alpha=regularization,
        max_iter=maxEpochs,
        random_state=seed,
        tol=1e-7
    )

    model.fit(XtrZ, YtrZ)

    Yhat_trZ = model.predict(XtrZ)
    Yhat_vaZ = model.predict(XvaZ)
    Yhat_teZ = model.predict(XteZ)

    Yhat_tr = scaler_Y.inverse_transform(Yhat_trZ.reshape(-1, 1)).ravel()
    Yhat_va = scaler_Y.inverse_transform(Yhat_vaZ.reshape(-1, 1)).ravel()
    Yhat_te = scaler_Y.inverse_transform(Yhat_teZ.reshape(-1, 1)).ravel()

    return Yhat_tr, Yhat_va, Yhat_te


# =========================
# Load data
# =========================
df = pd.read_excel(specFile, header=None)
df = df.apply(pd.to_numeric, errors="coerce")
data = df.to_numpy()

# MATLAB: waveband_all = data(1, 2:end)
waveband_all = data[0, 1:]

# MATLAB: reflectance = data(targetRows, 2:end)
target_row_indices = np.array(targetRows) - 1
reflectance = data[target_row_indices, 1:]


# =========================
# Extract selected wavelengths
# =========================
ridx = []
matched_wavelengths = []

for wb in wavebands_of_interest:
    matches = np.where(np.isclose(waveband_all, wb, atol=1e-6))[0]

    if len(matches) > 0:
        idx_match = matches[0]
    else:
        idx_match = np.nanargmin(np.abs(waveband_all - wb))
        print(
            f"Exact wavelength {wb:.4f} not found. "
            f"Using nearest wavelength {waveband_all[idx_match]:.4f}"
        )

    ridx.append(idx_match)
    matched_wavelengths.append(waveband_all[idx_match])

ridx = np.array(ridx)
matched_wavelengths = np.array(matched_wavelengths)

print("\nRequested wavelengths:")
print(wavebands_of_interest)

print("\nMatched wavelengths used:")
print(matched_wavelengths)

# samples x selected wavelengths
X_base = reflectance[:, ridx]


# =========================
# Main loop over target columns
# =========================
nTargets = len(targetCols)
heatMatrix = np.full((nTargets, 8), np.nan)

tGlobal = time.time()

for t in range(nTargets):

    targetCol = targetCols[t]
    targetName = targetLabels[t]

    print("\n====================================================")
    print(f"Processing targetCol = {targetCol} ({targetName})")
    print("====================================================")

    # MATLAB: Y_all = data(targetRows, targetCol)
    Y_all = data[target_row_indices, targetCol - 1]

    # Remove rows with NaN in X or Y
    valid_rows = np.all(np.isfinite(X_base), axis=1) & np.isfinite(Y_all)

    X = X_base[valid_rows, :]
    y = Y_all[valid_rows].ravel()

    N = X.shape[0]

    print(f"Usable samples: {N}")
    print(f"Selected wavelengths: {X.shape[1]}")

    if N < 10:
        print(f"Not enough usable samples for {targetName}. Skipping...")
        continue

    results = []
    tTarget = time.time()

    # =========================
    # Repeated ANN runs
    # =========================
    for run in range(1, numRuns + 1):

        seed = random.randint(1, 1_000_000)
        np.random.seed(seed)
        random.seed(seed)

        idx = np.random.permutation(N)

        # 70 / 15 / 15 split
        nTr = round(trainRatio * N)
        nTe = round(testRatio * N)
        nVa = N - nTr - nTe

        if nTr < 2 or nTe < 2 or nVa < 2:
            raise ValueError("Train/test/validation split is too small.")

        iTr = idx[:nTr]
        iTe = idx[nTr:nTr + nTe]
        iVa = idx[nTr + nTe:]

        Xtr = X[iTr, :]
        Ytr = y[iTr]

        Xte = X[iTe, :]
        Yte = y[iTe]

        Xva = X[iVa, :]
        Yva = y[iVa]

        Yhat_tr, Yhat_va, Yhat_te = train_ann_one(
            Xtr, Ytr, Xva, Yva, Xte, seed
        )

        # Metrics
        r_tr = corr_safe(Ytr, Yhat_tr)
        r_te = corr_safe(Yte, Yhat_te)
        r_va = corr_safe(Yva, Yhat_va)

        e_tr = nrmse(Ytr, Yhat_tr)
        e_te = nrmse(Yte, Yhat_te)
        e_va = nrmse(Yva, Yhat_va)

        # Overall metrics
        Y_all_true = np.concatenate([Ytr, Yte, Yva])
        Y_all_pred = np.concatenate([Yhat_tr, Yhat_te, Yhat_va])

        r_all = corr_safe(Y_all_true, Y_all_pred)
        e_all = nrmse(Y_all_true, Y_all_pred)

        # Generalization score
        gap_penalty = abs(r_tr - r_te) + abs(r_tr - r_va)

        score = (
            0.45 * r_te +
            0.45 * r_va +
            0.10 * r_all -
            0.25 * gap_penalty
        )

        results.append({
            "seed": seed,
            "r_tr": r_tr,
            "r_te": r_te,
            "r_va": r_va,
            "r_all": r_all,
            "e_tr": e_tr,
            "e_te": e_te,
            "e_va": e_va,
            "e_all": e_all,
            "score": score
        })

        print(
            f"Target {targetName} | Run {run:3d}/{numRuns:3d} | "
            f"Seed {seed:8d} | reg={regularization:.3f} | "
            f"r_tr={r_tr:.4f} | r_te={r_te:.4f} | "
            f"r_va={r_va:.4f} | gap={gap_penalty:.4f} | score={score:.4f}"
        )

    targetTime = time.time() - tTarget
    print(f"\nTime for {targetName}: {targetTime:.2f} seconds")

    # =========================
    # Sort runs and keep best runs
    # =========================
    results_sorted = sorted(results, key=lambda x: x["score"], reverse=True)

    topK = min(numTopRuns, numRuns)
    topResults = results_sorted[:topK]
    bestRun = results_sorted[0]

    print(f"\nBest single run for {targetName}:")
    print(f"Seed       -> {bestRun['seed']}")
    print(f"Train      -> r = {bestRun['r_tr']:.4f}, NRMSE = {bestRun['e_tr']:.4f}")
    print(f"Test       -> r = {bestRun['r_te']:.4f}, NRMSE = {bestRun['e_te']:.4f}")
    print(f"Validation -> r = {bestRun['r_va']:.4f}, NRMSE = {bestRun['e_va']:.4f}")
    print(f"Overall    -> r = {bestRun['r_all']:.4f}, NRMSE = {bestRun['e_all']:.4f}")

    # =========================
    # Average over best runs
    # =========================
    avg_r_tr = np.mean([r["r_tr"] for r in topResults])
    avg_r_te = np.mean([r["r_te"] for r in topResults])
    avg_r_va = np.mean([r["r_va"] for r in topResults])
    avg_r_all = np.mean([r["r_all"] for r in topResults])

    avg_e_tr = np.mean([r["e_tr"] for r in topResults])
    avg_e_te = np.mean([r["e_te"] for r in topResults])
    avg_e_va = np.mean([r["e_va"] for r in topResults])
    avg_e_all = np.mean([r["e_all"] for r in topResults])

    print("\n====================================================")
    print(f"Average over BEST {topK} generalizing runs for {targetName}")
    print(f"Train      -> r = {avg_r_tr:.4f}, NRMSE = {avg_e_tr:.4f}")
    print(f"Test       -> r = {avg_r_te:.4f}, NRMSE = {avg_e_te:.4f}")
    print(f"Validation -> r = {avg_r_va:.4f}, NRMSE = {avg_e_va:.4f}")
    print(f"Overall    -> r = {avg_r_all:.4f}, NRMSE = {avg_e_all:.4f}")
    print("====================================================")

    heatMatrix[t, :] = [
        avg_r_tr, avg_e_tr,
        avg_r_te, avg_e_te,
        avg_r_va, avg_e_va,
        avg_r_all, avg_e_all
    ]


# =========================
# Total time
# =========================
totalTime = time.time() - tGlobal
print(f"\nTotal time for all targets: {totalTime:.2f} seconds")


# =========================
# Save summary
# =========================
metricLabels = [
    "r_train", "NRMSE_train",
    "r_test", "NRMSE_test",
    "r_valid", "NRMSE_valid",
    "r_all", "NRMSE_all"
]

summary_df = pd.DataFrame(
    heatMatrix,
    index=targetLabels,
    columns=metricLabels
)

summary_df.to_excel("ANN_bottom_improved_summary.xlsx")
print("\nSaved summary to ANN_bottom_improved_summary.xlsx")


# =========================
# Heatmap
# =========================
plt.figure(figsize=(12, 5))
plt.imshow(heatMatrix, aspect="auto")
plt.colorbar(label="Metric value")

plt.xticks(np.arange(len(metricLabels)), metricLabels, rotation=45, ha="right")
plt.yticks(np.arange(len(targetLabels)), targetLabels)

for i in range(heatMatrix.shape[0]):
    for j in range(heatMatrix.shape[1]):
        plt.text(
            j,
            i,
            f"{heatMatrix[i, j]:.4f}",
            ha="center",
            va="center",
            color="black"
        )


plt.tight_layout()
plt.savefig("ANN_bottom_improved_heatmap.png", dpi=300)
plt.show()
