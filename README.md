# Maverick: Real-Time Evasion Detection in Tree Ensemble Automotive Intrusion Detection Systems

This repository contains the code and experiments for the paper titled **"Fast Evasion Detection & Alert Management in Tree-Ensemble-Based Intrusion Detection Systems"**, published in the Proceedings of the IEEE 36th International Conference on Tools with Artificial Intelligence (ICTAI), 2024.

## Dependencies

The following core libraries and tools are required for this project:

- [**VoTE**](https://github.com/john-tornblom/VoTE)
- [**OC-Score**](https://github.com/laudv/ocscore)

Additional Python packages:

- `pandas`
- `numpy`
- `tqdm`
- `functools`
- `scikit-learn`
- `xgboost`
- `pickle`
- `plotly`
- `collections`
- `imbalanced-learn`

## Repository Structure

All experiments and analyses are provided as Jupyter Notebooks for interactive, step-by-step execution. Each notebook is annotated with explanations and visualization cells to help you follow the methodology and results.

## Experimental Notes

The **evasion detection performance results** are located in folders with the prefix **`K-Fold`**, while the **pre-evasion signature detection results** are found in folders with the prefix **`Post-Detection`**. These folders also contain experiments and results related to **Sensitivity to Variations in Traffic Compositions**.

## Contact

For any questions, feel free to send me a message at [valency.colaco@liu.se](mailto:valency.colaco@liu.se)
