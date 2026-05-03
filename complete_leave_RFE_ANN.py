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
specFile = "completed_leave.xlsx"

# MATLAB rows 468:472
targetRows = [469, 470, 471, 472, 473]
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


# =========================
# Target-specific selected wavelengths
# =========================
wavebands_by_target = [
    # pH
    np.array([
        390.57, 391.89, 393.21, 402.42, 415.6, 418.23,
        430.1, 579.85, 659.92, 670.62, 673.3, 700.1,
        712.17, 732.31, 733.65, 741.71, 749.78, 755.16,
        791.52, 800.96, 841.47, 846.88, 879.37, 883.43,
        914.63, 922.78, 996.29, 1000.38, 1003.11, 1005.84
    ]),

    # EC
    np.array([
        391.89, 393.21, 405.06, 415.6, 435.38, 436.7,
        456.51, 489.58, 562.56, 563.89, 575.86, 577.19,
        586.51, 639.87, 666.61, 669.29, 673.3, 686.69,
        721.56, 724.25, 728.28, 736.34, 740.37, 883.43,
        967.66, 971.75, 985.38, 996.29, 1000.38, 1003.11
    ]),

    # NO3
    np.array([
        391.89, 407.69, 431.42, 494.87, 517.4, 546.61,
        551.92, 555.91, 557.24, 570.54, 610.5, 669.29,
        680.0, 696.07, 725.59, 726.93, 728.28, 729.62,
        732.31, 733.65, 734.99, 740.37, 741.71, 752.47,
        764.58, 807.7, 819.85, 883.43, 954.05, 1007.2
    ]),

    # Ca
    np.array([
        390.57, 391.89, 394.52, 395.84, 397.16, 398.47,
        399.79, 405.06, 407.69, 439.34, 468.4, 494.87,
        545.28, 607.83, 614.5, 653.23, 678.66, 681.34,
        696.07, 722.91, 728.28, 736.34, 749.78, 751.12,
        755.16, 883.43, 921.42, 971.75, 997.65, 1005.84
    ]),

    # Brix
    np.array([
        390.57, 391.89, 393.21, 395.84, 418.23, 436.7,
        526.69, 566.55, 585.18, 658.58, 678.66, 681.34,
        689.37, 692.05, 694.73, 708.14, 717.54, 721.56,
        722.91, 730.96, 736.34, 744.4, 757.85, 790.17,
        906.49, 922.78, 952.69, 986.74, 1003.11, 1008.57
    ])
]


# =========================
# Metric functions
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
        return 0.0

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

# MATLAB column 2 = Python index 1
waveband_all = data[:, 1]

# MATLAB columns 3:end = Python index 2 onward
reflectance = data[:, 2:]


# =========================
# Main loop
# =========================
nTargets = len(targetRows)
heatMatrix = np.full((nTargets, 8), np.nan)

tGlobal = time.time()

for t in range(nTargets):

    targetRow = targetRows[t]
    targetName = targetLabels[t]
    wavebands_of_interest = wavebands_by_target[t]

    print("\n====================================================")
    print(f"Processing targetRow = {targetRow} ({targetName})")
    print("====================================================")

    # =========================
    # Extract target-specific wavelengths
    # =========================
    ridx = []

    for wb in wavebands_of_interest:
        matches = np.where(np.isclose(waveband_all, wb, atol=1e-6))[0]

        if len(matches) == 0:
            nearest_idx = np.nanargmin(np.abs(waveband_all - wb))
            nearest_val = waveband_all[nearest_idx]

            if abs(nearest_val - wb) <= 0.05:
                print(f"Using nearest wavelength {nearest_val:.4f} for requested {wb:.4f}")
                ridx.append(nearest_idx)
            else:
                raise ValueError(
                    f"For target {targetName}, wavelength not found: {wb}"
                )
        else:
            ridx.append(matches[0])

    ridx = np.array(ridx)

    X_base = reflectance[ridx, :]

    # IMPORTANT:
    # MATLAB row 468 = Python index 467
    Y_all = data[targetRow - 1, 2:]

    # =========================
    # Remove columns with NaN in X or Y
    # =========================
    valid_cols = np.all(np.isfinite(X_base), axis=0) & np.isfinite(Y_all)

    X_all = X_base[:, valid_cols]
    Y_use = Y_all[valid_cols]

    # Final format: samples in rows
    X = X_all.T
    y = Y_use.ravel()

    N = X.shape[0]

    print(f"Usable samples: {N}")
    print(f"Number of selected wavelengths for {targetName}: {len(wavebands_of_interest)}")

    if N < 10:
        raise ValueError(f"Not enough usable samples for {targetName}")

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

    # =========================
    # Compare summary groups
    # =========================
    groups = [1, 10, 25, 50, 100, 150]

    print(f"\n================ Performance Summary for {targetName} ================")

    for g in groups:
        k = min(g, numRuns)
        subset = results_sorted[:k]

        print(f"\nAverage over BEST {k} runs")
        print(
            f"Train      -> r = {np.mean([r['r_tr'] for r in subset]):.4f}, "
            f"NRMSE = {np.mean([r['e_tr'] for r in subset]):.4f}"
        )
        print(
            f"Test       -> r = {np.mean([r['r_te'] for r in subset]):.4f}, "
            f"NRMSE = {np.mean([r['e_te'] for r in subset]):.4f}"
        )
        print(
            f"Validation -> r = {np.mean([r['r_va'] for r in subset]):.4f}, "
            f"NRMSE = {np.mean([r['e_va'] for r in subset]):.4f}"
        )
        print(
            f"Overall    -> r = {np.mean([r['r_all'] for r in subset]):.4f}, "
            f"NRMSE = {np.mean([r['e_all'] for r in subset]):.4f}"
        )

    print("===========================================================")

    # Save to heatmap matrix
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

summary_df.to_excel("ANN_RFE_improved_summary.xlsx")
print("\nSaved summary to ANN_RFE_improved_summary.xlsx")


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
plt.savefig("ANN_RFE_improved_heatmap.png", dpi=300)
plt.show()