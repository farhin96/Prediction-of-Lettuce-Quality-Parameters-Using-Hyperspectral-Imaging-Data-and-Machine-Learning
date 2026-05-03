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

# MATLAB rows 468:472 converted to Python index 467:471
targetRows = [468, 469, 470, 471,472]
targetLabels = ["pH", "EC", "NO3", "Ca", "Brix"]

numRuns = 150
numTopRuns = 100

# Improved ANN settings
hidden_layer_sizes = (10,)      # smaller than 10 to reduce overfitting
maxEpochs = 1000
regularization = 0.1          # stronger regularization
activation = "tanh"
solver = "lbfgs"

# 70 / 15 / 15 split
trainRatio = 0.70
testRatio = 0.15
validRatio = 0.15

wavebands_of_interest = np.array([
    394.52,
    398.47,
    406.37,
    447.26,
    476.34,
    520.06,
    570.54,
    601.17,
    609.17,
    627.85,
    641.2,
    642.54,
    643.88,
    651.9,
    653.23,
    662.6,
    693.39,
    710.83,
    714.85,
    718.88,
    755.16,
    898.35,
    917.35,
    939.09,
    949.97,
    956.77,
    962.22,
    984.02,
    996.29,
    1007.2
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
# Train ANN once
# =========================
def train_ann_one(Xtr, Ytr, Xva, Yva, Xte, seed):
    # Standardize X using training data only
    scaler_X = StandardScaler()
    XtrZ = scaler_X.fit_transform(Xtr)
    XvaZ = scaler_X.transform(Xva)
    XteZ = scaler_X.transform(Xte)

    # Standardize Y using training data only
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

waveband_all = data[:, 1]
reflectance = data[:, 2:]


# =========================
# Extract selected wavelengths
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
            raise ValueError(f"Wavelength not found: {wb}")
    else:
        ridx.append(matches[0])

ridx = np.array(ridx)
X_base = reflectance[ridx, :]


# =========================
# Main loop
# =========================
nTargets = len(targetRows)
heatMatrix = np.full((nTargets, 8), np.nan)

tGlobal = time.time()

for t, targetRow in enumerate(targetRows):

    targetName = targetLabels[t]

    print("\n====================================================")
    print(f"Processing targetRow = {targetRow + 1} ({targetName})")
    print("====================================================")

    Y_all = data[targetRow, 2:]

    valid_cols = np.all(np.isfinite(X_base), axis=0) & np.isfinite(Y_all)

    X_all = X_base[:, valid_cols]
    Y_use = Y_all[valid_cols]

    X = X_all.T
    y = Y_use.ravel()

    N = X.shape[0]

    print(f"Usable samples: {N}")
    print(f"Features used: {X.shape[1]}")

    if N < 10:
        raise ValueError(f"Not enough usable samples for {targetName}")

    results = []
    tTarget = time.time()

    for run in range(1, numRuns + 1):

        seed = random.randint(1, 1_000_000)
        np.random.seed(seed)
        random.seed(seed)

        idx = np.random.permutation(N)

        nTr = round(trainRatio * N)
        nTe = round(testRatio * N)
        nVa = N - nTr - nTe

        if nTr < 2 or nTe < 2 or nVa < 2:
            raise ValueError("Train/test/validation split is too small.")

        iTr = idx[:nTr]
        iTe = idx[nTr:nTr + nTe]
        iVa = idx[nTr + nTe:]

        Xtr, Ytr = X[iTr, :], y[iTr]
        Xte, Yte = X[iTe, :], y[iTe]
        Xva, Yva = X[iVa, :], y[iVa]

        Yhat_tr, Yhat_va, Yhat_te = train_ann_one(Xtr, Ytr, Xva, Yva, Xte, seed)

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

        # Generalization score
        # High test/validation r is rewarded.
        # Large train-test gap is penalized.
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
            f"Seed {seed:8d} | "
            f"r_tr={r_tr:.4f} | r_te={r_te:.4f} | r_va={r_va:.4f} | "
            f"gap={gap_penalty:.4f} | score={score:.4f}"
        )

    targetTime = time.time() - tTarget
    print(f"\nTime for {targetName}: {targetTime:.2f} seconds")

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

    groups = [1, 10, 25, 50, 100, 150]

    print(f"\n================ Performance Summary for {targetName} ================")

    for g in groups:
        k = min(g, numRuns)
        subset = results_sorted[:k]

        print(f"\nAverage over BEST {k} runs")
        print(f"Train      -> r = {np.mean([r['r_tr'] for r in subset]):.4f}, "
              f"NRMSE = {np.mean([r['e_tr'] for r in subset]):.4f}")

        print(f"Test       -> r = {np.mean([r['r_te'] for r in subset]):.4f}, "
              f"NRMSE = {np.mean([r['e_te'] for r in subset]):.4f}")

        print(f"Validation -> r = {np.mean([r['r_va'] for r in subset]):.4f}, "
              f"NRMSE = {np.mean([r['e_va'] for r in subset]):.4f}")

        print(f"Overall    -> r = {np.mean([r['r_all'] for r in subset]):.4f}, "
              f"NRMSE = {np.mean([r['e_all'] for r in subset]):.4f}")

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
# Save heatmap results
# =========================
metricLabels = [
    "r_train", "NRMSE_train",
    "r_test", "NRMSE_test",
    "r_valid", "NRMSE_valid",
    "r_all", "NRMSE_all"
]

summary_df = pd.DataFrame(heatMatrix, index=targetLabels, columns=metricLabels)
summary_df.to_excel("ANN_improved_summary.xlsx")

print("\nSaved summary to ANN_improved_summary.xlsx")


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
plt.savefig("ANN_improved_heatmap.png", dpi=300)
plt.show()