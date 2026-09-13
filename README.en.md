# Telecom X — customer churn modeling, part 2

[Español](README.md) · [Notebook](Challenge_Telecom_X_an%C3%A1lisis_de_evasi%C3%B3n_de_clientes_Parte_2.ipynb)

A classification exercise associated with the Telecom X challenge. It contains feature preparation, SMOTE oversampling, logistic regression and Random Forest training.

**Status: reproducibility pending.** The current repository cannot run the entire workflow from a clean session. Metrics written in the notebook are not presented here as verified results.

## Included code

- Reading `datos_tratados.csv`.
- Encoding with `pandas.get_dummies`.
- A 70/30 split with `random_state=42`.
- SMOTE applied to the training set.
- Scaling and training two classifiers.
- Evaluation and feature-importance code.

## Execution blockers

1. Supply `datos_tratados.csv` with documented provenance and transformations. It is not versioned, and the part 1 notebook does not export it.
2. Correct references to `df`; the notebook initially loads `df_telecom`.
3. Import `accuracy_score` and `confusion_matrix`, used during evaluation.
4. Consolidate splitting and preprocessing, currently redefined across cells.
5. Review `astype(int)`, which also converts continuous features and can discard decimals.
6. Run from a clean session and save traceable results before quoting accuracy, recall or overfitting.

These are code-review observations. This documentation update does not modify the notebook.

## Prepare an environment

```bash
git clone https://github.com/fabrizzio2901/Challenge-Telecom-X-an-lisis-de-evasi-n-de-clientes---Parte-2.git
cd Challenge-Telecom-X-an-lisis-de-evasi-n-de-clientes---Parte-2
python -m venv .venv
```

Activate `.venv` using `.\.venv\Scripts\Activate.ps1` in PowerShell or `source .venv/bin/activate` on macOS/Linux. Then:

```bash
python -m pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn jupyterlab
python -m jupyterlab
```

Open the notebook linked above. These commands prepare an environment; they do not resolve the blockers or represent a validated execution. Dependencies are not pinned.

## Credits and scope

An Alura Telecom X challenge exercise. It is not a deployed model or evidence of a real reduction in churn. The connection to [the exploratory analysis](https://github.com/fabrizzio2901/TelecomX-Datos) needs a reproducible export of its input dataset.

## My contribution

I contributed to implementing this modeling exercise as part of my technical learning. It complements my full-stack development portfolio and retains the reproducibility issues described above.
