import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load Excel file
df = pd.read_excel("completed_leave.xlsx", header=None)

# Convert all values to numeric
df = df.apply(pd.to_numeric, errors="coerce")

# Remove rows with missing/non-numeric waveband values
df = df.dropna(subset=[1])

# Extract waveband and reflectance values
waveband = df.iloc[:, 1].to_numpy()
reflectance = df.iloc[:, 2:].to_numpy()

# Filter waveband <= 1009 nm
valid_rows = waveband <= 1009
waveband = waveband[valid_rows]
reflectance = reflectance[valid_rows, :]

# Plot reflectance spectra
plt.figure(1)
plt.plot(waveband, reflectance, linewidth=1)
plt.xlabel("Waveband, [nm]")
plt.ylabel("Reflectance")
plt.grid(True)
plt.tight_layout()
plt.show()

# Compute First Derivative Reflectance
dR = np.diff(reflectance, axis=0) / np.diff(waveband)[:, None]
waveband_fdr = waveband[:-1]

# Plot FDR spectrum
plt.figure(2)
plt.plot(waveband_fdr, dR, linewidth=1)
plt.xlabel("Waveband, [nm]")
plt.ylabel("dR")
plt.grid(True)
plt.xlim([390, 1009])
plt.tight_layout()
plt.show()