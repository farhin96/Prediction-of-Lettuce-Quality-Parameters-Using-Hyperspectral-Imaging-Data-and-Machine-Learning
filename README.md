# Prediction-of-Lettuce-Quality-Parameters-Using-Hyperspectral-Imaging-Data-and-Machine-Learning

Hyperspectral Feature Selection and ANN Modeling

This folder contains Python scripts for hyperspectral reflectance analysis, feature selection, and ANN-based prediction of lettuce biochemical/nutrient traits.
The workflow includes three feature selection methods and one ANN modeling method:
FDR: First Derivative Reflectance;
PCA: Principal Component Analysis based wavelength selection;
RFE: Recursive Feature Elimination using Random Forest;
ANN: Artificial Neural Network regression model;
.The scripts are prepared for both complete leaf data and specific leaf-region data such as apex, middle, and bottom regions.

---
Files included



`complete_leave_FDR.py`	: Computes and plots first derivative reflectance from complete leaf spectra;
`complete_leave_PCA.py`	: Selects important wavelengths using PCA;
`complete_leave_RFE.py`	: Selects important wavelengths using RFE and Random Forest;
`complete_leave_FDR_ANN.py`	: Runs ANN using FDR-selected wavelengths;
`complete_leave_PCA_ANN.py`	: Runs ANN using PCA-selected wavelengths;
`complete_leave_RFE_ANN.py` :	Runs ANN using RFE-selected wavelengths;

Specific leaf-region (Apex, Middle, Bottom) scripts 


`specific_part_FDR.py`	: Computes and plots first derivative reflectance for a specific leaf region;
`specific_part_PCA.py`	: Selects important wavelengths from a specific leaf region using PCA;
`specific_part_RFE.py`	: Selects important wavelengths from a specific leaf region using RFE;
`specific_part_FDR_ANN.py`	: Runs ANN using FDR-selected wavelengths for a specific leaf region;
`specific_part_PCA_ANN.py`	: Runs ANN using PCA-selected wavelengths for a specific leaf region;
`specific_part_RFE_ANN.py`	: Runs ANN using RFE-selected wavelengths for a specific leaf region;

Data files


`completed_leave.xlsx`	: Complete leaf hyperspectral data;
`GH_1and2_phase_apex.xlsx`	: Apex region hyperspectral data;
`GH_1and2_phase_middle.xlsx`	: Middle region hyperspectral data;
`GH_1and2_phase_bottom.xlsx`	: Bottom region hyperspectral data;

---
Required Python packages

Install the required packages before running the scripts.
```bash
pip install numpy pandas matplotlib scipy scikit-learn openpyxl
```
---
Recommended folder setup

Keep all `.py` scripts and Excel files in the same folder.
```text
project_folder/
├── completed_leave.xlsx
├── GH_1and2_phase_apex.xlsx
├── GH_1and2_phase_middle.xlsx
├── GH_1and2_phase_bottom.xlsx
├── complete_leave_FDR.py
├── complete_leave_PCA.py
├── complete_leave_RFE.py
├── complete_leave_FDR_ANN.py
├── complete_leave_PCA_ANN.py
├── complete_leave_RFE_ANN.py
├── specific_part_FDR.py
├── specific_part_PCA.py
├── specific_part_RFE.py
├── specific_part_FDR_ANN.py
├── specific_part_PCA_ANN.py
└── specific_part_RFE_ANN.py
```
---
How to run the scripts

Open Command Prompt, PowerShell, Terminal, or Anaconda Prompt.
Move to the project folder:
```bash
cd path/to/project_folder
```
For example, on Windows:
```bash
cd D:\different_region\python
```
---
Complete leaf analysis

Step 1: Run feature selection
Run one of the following scripts depending on the feature selection method.

FDR
```bash
python complete_leave_FDR.py
```
This script calculates the first derivative of reflectance and plots the FDR spectra;


PCA
```bash
python complete_leave_PCA.py
```
This script performs PCA, calculates loading-based wavelength importance, and prints selected wavelengths.;


RFE
```bash
python complete_leave_RFE.py

```


This script performs Recursive Feature Elimination using Random Forest and prints selected wavelengths for each target variable.


---

Step 2: Run ANN modeling

After selecting wavelengths, run the corresponding ANN script.
```bash
python complete_leave_FDR_ANN.py
python complete_leave_PCA_ANN.py
python complete_leave_RFE_ANN.py
```
Each ANN script trains models for five target variables:
pH,
EC,
NO3,
Ca,
Brix,

The ANN scripts report:


Training correlation


Testing correlation


Validation correlation


Overall correlation


Training NRMSE


Testing NRMSE


Validation NRMSE


Overall NRMSE

---
Specific leaf-region analysis

The specific-region scripts are used for excel files such as:
`GH_1and2_phase_apex.xlsx`
`GH_1and2_phase_middle.xlsx`
`GH_1and2_phase_bottom.xlsx`
Before running these scripts, check the `specFile` variable inside the script.

Example:
```python
specFile = "GH_1and2_phase_bottom.xlsx"
```
Change it if needed:
```python
specFile = "GH_1and2_phase_apex.xlsx"
```
or:
```python
specFile = "GH_1and2_phase_middle.xlsx"
```
---
Step 1: Run feature selection for a specific leaf region

```bash
python specific_part_FDR.py
python specific_part_PCA.py
python specific_part_RFE.py
```
---
Step 2: Run ANN modeling for a specific leaf region

```bash
python specific_part_FDR_ANN.py
python specific_part_PCA_ANN.py
python specific_part_RFE_ANN.py
```
---
Important settings to check before running ANN

Open the ANN script and check these settings near the top.
```python
numRuns = 150
numTopRuns = 100

hidden_layer_sizes = (10,)
maxEpochs = 1000
regularization = 0.1
activation = "tanh"
solver = "lbfgs"

trainRatio = 0.70
testRatio = 0.15
validRatio = 0.15
```
Meaning:


`numRuns`: number of repeated random train/test/validation splits;


`numTopRuns`: number of best runs used for final average;


`hidden_layer_sizes`: ANN hidden layer structure;


`regularization`: controls overfitting;


`activation`: ANN activation function;


`solver`: optimization method;


`trainRatio`, `testRatio`, `validRatio`: data split ratio;

---
Updating selected wavelengths

The ANN scripts use a wavelength list such as:
```python
wavebands_of_interest = np.array([
    390.57, 391.89, 393.21
])
```
or a target-specific dictionary such as:
```python
selectedBands = {
    "pH": np.array([...]),
    "EC": np.array([...]),
    "NO3": np.array([...]),
    "Ca": np.array([...]),
    "Brix": np.array([...])
}
```
If you run PCA or RFE and get new selected wavelengths, copy those wavelengths into the corresponding ANN script before running the ANN model.

Suggested workflow

```text
1. Run feature selection
   - FDR, PCA, or RFE

2. Copy selected wavelengths into the related ANN script

3. Run ANN model

4. Compare train, test, validation, and overall performance

5. Use testing and validation results to judge model reliability
---
