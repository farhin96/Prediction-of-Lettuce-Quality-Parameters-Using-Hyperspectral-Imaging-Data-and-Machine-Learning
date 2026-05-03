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

targetRows = list(range(2, 110))          # MATLAB 2:109
targetCols = [465, 466, 467, 468, 469]   # MATLAB 465:469
targetLabels = ["pH", "EC", "NO3", "Ca", "Brix"]

numRuns = 150
numTopRuns = 100

hidden_layer_sizes = (10,)
maxEpochs = 1000
regularization = 0.1
activation = "tanh"
solver = "lbfgs"


# =========================
# Target-specific RFE wavelengths
# =========================
selectedBands = {
    "pH": np.array([
        390.57, 393.21, 395.84, 397.16, 407.69, 409.01,
        416.92, 419.55, 451.22, 481.63, 547.93, 549.26,
        602.50, 630.52, 713.51, 720.22, 724.25, 730.96,
        749.78, 819.85, 844.17, 895.63, 910.56, 948.61,
        954.05, 979.93, 986.74, 996.29, 1000.38, 1005.84
    ]),

    "EC": np.array([
        390.57, 391.89, 399.79, 482.96, 534.65, 547.93,
        549.26, 550.59, 553.25, 559.90, 579.85, 606.50,
        646.55, 659.92, 685.35, 690.71, 714.85, 716.19,
        721.56, 725.59, 733.65, 745.75, 792.87, 888.85,
        901.06, 959.50, 996.29, 999.02, 1001.75, 1005.84
    ]),

    "NO3": np.array([
        390.57, 391.89, 405.06, 415.60, 452.54, 504.15,
        517.40, 533.33, 558.57, 562.56, 647.89, 662.60,
        704.12, 718.88, 729.62, 733.65, 734.99, 740.37,
        799.61, 895.63, 898.35, 935.01, 945.89, 955.41,
        959.50, 964.94, 985.38, 999.02, 1001.75, 1005.84
    ]),

    "Ca": np.array([
        390.57, 391.89, 393.21, 394.52, 395.84, 398.47,
        401.11, 418.23, 439.34, 558.57, 569.21, 663.93,
        682.68, 692.05, 694.73, 722.91, 730.96, 740.37,
        757.85, 759.20, 799.61, 800.96, 891.57, 907.84,
        913.27, 952.69, 956.77, 982.65, 989.47, 996.29
    ]),

    "Brix": np.array([
        390.57, 391.89, 397.16, 401.11, 402.42, 403.74,
        409.01, 411.64, 431.42, 435.38, 501.50, 563.89,
        610.50, 634.52, 637.20, 670.62, 689.37, 701.44,
        720.22, 722.91, 730.96, 741.71, 760.54, 940.45,
        966.30, 977.20, 982.65, 992.20, 1005.84, 1008.57
    ])
}

# =========================
# Metric helpers
# =========================
def nrmse(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return rmse / (np.max(y_true) - np.min(y_true) + np.finfo(float).eps)


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
    return 0.0 if np.isnan(r) else r


# =========================
# ANN function
# =========================
def train_ann_one(Xtr, Ytr, Xva, Yva, Xte, seed):
    scaler_X = StandardScaler()
    XtrZ = scaler_X.fit_transform(Xtr)
    XvaZ = scaler_X.transform(Xva)
    XteZ = scaler_X.transform(Xte)

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

    Yhat_tr = scaler_Y.inverse_transform(model.predict(XtrZ).reshape(-1, 1)).ravel()
    Yhat_va = scaler_Y.inverse_transform(model.predict(XvaZ).reshape(-1, 1)).ravel()
    Yhat_te = scaler_Y.inverse_transform(model.predict(XteZ).reshape(-1, 1)).ravel()

    return Yhat_tr, Yhat_va, Yhat_te


# =========================
# Load data
# =========================
df = pd.read_excel(specFile, header=None)
df = df.apply(pd.to_numeric, errors="coerce")
data = df.to_numpy()

target_row_indices = np.array(targetRows) - 1

waveband_all = data[0, 1:]
reflectance_all = data[target_row_indices, 1:]


# =========================
# Main loop
# =========================
heatMatrix = np.full((len(targetCols), 8), np.nan)

tGlobal = time.time()

for t, targetName in enumerate(targetLabels):

    targetCol = targetCols[t]
    bands_this_target = selectedBands[targetName]

    print("\n====================================================")
    print(f"Processing {targetName} using targetCol = {targetCol}")
    print("====================================================")

    # Find selected wavelength indices
    ridx = []

    for wb in bands_this_target:
        matches = np.where(np.isclose(waveband_all, wb, atol=1e-6))[0]

        if len(matches) > 0:
            ridx.append(matches[0])
        else:
            nearest_idx = np.nanargmin(np.abs(waveband_all - wb))
            nearest_val = waveband_all[nearest_idx]
            print(
                f"Exact wavelength {wb:.4f} not found for {targetName}. "
                f"Using nearest {nearest_val:.4f}"
            )
            ridx.append(nearest_idx)

    ridx = np.array(ridx)

    X_base = reflectance_all[:, ridx]
    Y_all = data[target_row_indices, targetCol - 1]

    valid_rows = np.all(np.isfinite(X_base), axis=1) & np.isfinite(Y_all)

    X = X_base[valid_rows, :]
    y = Y_all[valid_rows].ravel()

    N = X.shape[0]

    print(f"Selected wavelengths for {targetName}: {X.shape[1]}")
    print(f"Usable samples: {N}")

    if N < 10:
        print(f"Not enough usable samples for {targetName}. Skipping...")
        continue

    results = []
    tTarget = time.time()

    for run in range(1, numRuns + 1):

        seed = random.randint(1, 1_000_000)
        np.random.seed(seed)
        random.seed(seed)

        idx = np.random.permutation(N)

        nTr = round(0.8 * N)
        nTe = round(0.1 * N)
        nVa = N - nTr - nTe

        iTr = idx[:nTr]
        iTe = idx[nTr:nTr + nTe]
        iVa = idx[nTr + nTe:]

        Xtr, Ytr = X[iTr, :], y[iTr]
        Xte, Yte = X[iTe, :], y[iTe]
        Xva, Yva = X[iVa, :], y[iVa]

        Yhat_tr, Yhat_va, Yhat_te = train_ann_one(
            Xtr, Ytr, Xva, Yva, Xte, seed
        )

        r_tr = corr_safe(Ytr, Yhat_tr)
        r_te = corr_safe(Yte, Yhat_te)
        r_va = corr_safe(Yva, Yhat_va)

        e_tr = nrmse(Ytr, Yhat_tr)
        e_te = nrmse(Yte, Yhat_te)
        e_va = nrmse(Yva, Yhat_va)

        Y_all_true = np.concatenate([Ytr, Yte, Yva])
        Y_all_pred = np.concatenate([Yhat_tr, Yhat_te, Yhat_va])

        r_all = corr_safe(Y_all_true, Y_all_pred)
        e_all = nrmse(Y_all_true, Y_all_pred)

        score = 2 * (r_va * r_te) / (r_va + r_te + np.finfo(float).eps)

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
            f"Seed {seed:10d} | r_tr={r_tr:.4f} | "
            f"r_te={r_te:.4f} | r_va={r_va:.4f} | score={score:.4f}"
        )

    print(f"\nTime for {targetName}: {time.time() - tTarget:.2f} seconds")

    results_sorted = sorted(results, key=lambda x: x["score"], reverse=True)

    topK = min(numTopRuns, numRuns)
    topResults = results_sorted[:topK]
    bestRun = results_sorted[0]

    print(f"\nBest single run for {targetName}:")
    print(f"Train      -> r = {bestRun['r_tr']:.4f}, NRMSE = {bestRun['e_tr']:.4f}")
    print(f"Test       -> r = {bestRun['r_te']:.4f}, NRMSE = {bestRun['e_te']:.4f}")
    print(f"Validation -> r = {bestRun['r_va']:.4f}, NRMSE = {bestRun['e_va']:.4f}")
    print(f"Overall    -> r = {bestRun['r_all']:.4f}, NRMSE = {bestRun['e_all']:.4f}")

    avg_r_tr = np.mean([r["r_tr"] for r in topResults])
    avg_r_te = np.mean([r["r_te"] for r in topResults])
    avg_r_va = np.mean([r["r_va"] for r in topResults])
    avg_r_all = np.mean([r["r_all"] for r in topResults])

    avg_e_tr = np.mean([r["e_tr"] for r in topResults])
    avg_e_te = np.mean([r["e_te"] for r in topResults])
    avg_e_va = np.mean([r["e_va"] for r in topResults])
    avg_e_all = np.mean([r["e_all"] for r in topResults])

    print("\n====================================================")
    print(f"Average over BEST {topK} runs for {targetName}")
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


print(f"\nTotal time for all targets: {time.time() - tGlobal:.2f} seconds")


# =========================
# Save results
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

summary_df.to_excel("ANN_middle_target_specific_RFE_summary.xlsx")
print("\nSaved summary to ANN_middle_target_specific_RFE_summary.xlsx")


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
        plt.text(j, i, f"{heatMatrix[i, j]:.4f}",
                 ha="center", va="center", color="black")


plt.tight_layout()
plt.savefig("ANN_middle_target_specific_RFE_heatmap.png", dpi=300)
plt.show()
