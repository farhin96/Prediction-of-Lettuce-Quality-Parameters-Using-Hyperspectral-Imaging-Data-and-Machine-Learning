import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load numeric spectral data
data = pd.read_excel("GH_1and2_phase_bottom.xlsx", header=None)
data = data.apply(pd.to_numeric, errors="coerce").to_numpy()

# Extract waveband and reflectance values
waveband = data[0, 1:]          # MATLAB: data(1, 2:end)
reflectance = data[1:, 1:]      # MATLAB: data(2:end, 2:end)

# Filter wavelengths up to 1009 nm
valid_cols = waveband <= 1009
waveband = waveband[valid_cols]
reflectance = reflectance[:, valid_cols]

# Plot reflectance spectra
plt.figure(1)
plt.plot(waveband, reflectance.T, linewidth=1)
plt.xlabel("Waveband, [nm]")
plt.ylabel("Reflectance")
plt.grid(True)
plt.tight_layout()
plt.show()

# =========================
# Compute First Derivative Reflectance
# =========================
FDR = np.diff(reflectance, axis=1) / np.diff(waveband)

# Adjust waveband for FDR using midpoint
waveband_FDR = (waveband[:-1] + waveband[1:]) / 2

# =========================
# Plot FDR
# =========================
plt.figure(2)
plt.plot(waveband_FDR, FDR.T, linewidth=1)
plt.xlabel("Waveband, [nm]")
plt.ylabel("FDR (dR/dλ)")
plt.title("First Derivative of Reflectance")
plt.grid(True)
plt.tight_layout()
plt.show()