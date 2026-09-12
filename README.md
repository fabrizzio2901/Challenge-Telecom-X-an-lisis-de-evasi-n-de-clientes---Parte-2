# Telecom X — modelado de cancelación de clientes, parte 2

[English](README.en.md) · [Notebook](Challenge_Telecom_X_an%C3%A1lisis_de_evasi%C3%B3n_de_clientes_Parte_2.ipynb)

Ejercicio de clasificación asociado al desafío Telecom X. Contiene preparación de variables, sobremuestreo con SMOTE y entrenamiento de regresión logística y Random Forest.

**Estado: reproducción pendiente.** El repositorio actual no permite ejecutar todo el flujo desde una sesión limpia. Las métricas escritas en el notebook no se presentan aquí como resultados verificados.

## Código incluido

- Lectura de `datos_tratados.csv`.
- Codificación con `pandas.get_dummies`.
- División 70/30 con `random_state=42`.
- SMOTE sobre el conjunto de entrenamiento.
- Escalado y entrenamiento de los dos clasificadores.
- Código de evaluación e importancia de variables.

## Qué falta para ejecutarlo

1. Proporcionar `datos_tratados.csv` con procedencia y transformación documentadas. No está versionado y el notebook de la parte 1 no incluye su exportación.
2. Corregir referencias a `df`: el cuaderno carga inicialmente `df_telecom`.
3. Importar `accuracy_score` y `confusion_matrix`, utilizados en la evaluación.
4. Unificar el flujo de partición y preprocesamiento, que se redefine en varias celdas.
5. Revisar `astype(int)`, que también convierte variables continuas y puede perder sus decimales.
6. Ejecutar todo desde cero y guardar resultados trazables antes de citar exactitud, recall o sobreajuste.

Estas son observaciones de código; esta actualización de documentación no modifica el notebook.

## Preparar el entorno

```bash
git clone https://github.com/fabrizzio2901/Challenge-Telecom-X-an-lisis-de-evasi-n-de-clientes---Parte-2.git
cd Challenge-Telecom-X-an-lisis-de-evasi-n-de-clientes---Parte-2
python -m venv .venv
```

Activa `.venv` con `.\.venv\Scripts\Activate.ps1` en PowerShell o `source .venv/bin/activate` en macOS/Linux. Después:

```bash
python -m pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn jupyterlab
python -m jupyterlab
```

Abre el notebook enlazado al inicio. Estos comandos preparan un entorno; no resuelven los bloqueos descritos ni constituyen una ejecución validada. No hay versiones fijadas.

## Créditos y alcance

Ejercicio del desafío Telecom X de Alura. No es un modelo desplegado ni evidencia de reducción real de cancelaciones. La relación con [la parte exploratoria](https://github.com/fabrizzio2901/TelecomX-Datos) debe completarse mediante una exportación reproducible del conjunto de entrada.

