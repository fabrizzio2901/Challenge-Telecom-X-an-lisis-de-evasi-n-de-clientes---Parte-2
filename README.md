# Challenge-Telecom-X-an-lisis-de-evasi-n-de-clientes---Parte-2

# Telecom X - Fase 2: Predicción de Cancelación de Clientes 

## 1. Propósito del Proyecto
El objetivo principal de este análisis es desarrollar un modelo de Machine Learning capaz de predecir la probabilidad de que un cliente cancele sus servicios (**Churn**) en la empresa Telecom X. 

Mediante el análisis de variables demográficas, servicios contratados y métricas financieras, buscamos identificar patrones de comportamiento que permitan a la empresa pasar de una postura reactiva a una estrategia de retención preventiva, optimizando así sus recursos y aumentando la lealtad del cliente.

---

## 2. Estructura del Proyecto
El proyecto está organizado de la siguiente manera:

* `TelecomX_Predictivo.ipynb`: Cuaderno principal (Google Colab) con todo el flujo de datos y modelado.
* `datos_tratados.csv`: Conjunto de datos limpio y procesado derivado de la Fase 1.
* `visualizaciones/`: Carpeta que contiene los gráficos generados (Matriz de confusión, importancia de variables, etc.).
* `README.md`: Descripción general del proyecto.

---

## 3. Preparación y Modelado de Datos

### Clasificación de Variables
Se identificaron y separaron las variables para su tratamiento específico:
* **Numéricas:** `tenure` (antigüedad), `Charges.Monthly`, `Charges.Total`, entre otras.
* **Categóricas:** `InternetService`, `Contract`, `PaymentMethod`, etc.

### Preprocesamiento
1.  **Codificación:** Se utilizó *One-Hot Encoding* (`get_dummies`) para transformar las variables categóricas en binarias, permitiendo su procesamiento matemático.
2.  **Normalización:** Se aplicó `StandardScaler` a los modelos sensibles a la escala (como Regresión Logística), asegurando que variables con rangos grandes (ej. Gasto Total) no dominaran injustamente la predicción.
3.  **Balanceo de Clases:** Debido al desequilibrio en los datos (pocos casos de Churn), se utilizó la técnica de sobremuestreo para equilibrar las clases y mejorar la detección de desertores.

### Entrenamiento y Prueba
Los datos se dividieron en:
* **70% Entrenamiento:** Para que los modelos aprendan los patrones.
* **30% Prueba:** Para evaluar el desempeño final con datos que el modelo nunca ha visto.

---

## 4. Análisis Exploratorio de Datos (EDA) e Insights

Durante el análisis exploratorio se obtuvieron hallazgos críticos que guiaron el modelado:

* **Antigüedad (Tenure):** Existe una fuerte correlación negativa con la cancelación; los clientes en su primer año de servicio presentan el riesgo más alto.
* **Contratos:** Los usuarios con contratos "Mes a Mes" tienen una tasa de fuga significativamente superior a quienes poseen contratos anuales o bianuales.
* **Servicios:** Los clientes con Fibra Óptica presentan una tendencia inusual a la cancelación, lo que sugiere áreas de oportunidad en la calidad o costo de dicho servicio.



---

## 5. Modelización y Resultados
Se compararon dos modelos principales:

1.  **Regresión Logística:** Seleccionada como el modelo más robusto debido a su alto **Recall (80%)**, permitiendo identificar a la gran mayoría de los clientes en riesgo.
2.  **Random Forest:** Mostró una buena precisión general, pero presentó signos de *overfitting*, memorizando los datos de entrenamiento sin generalizar eficientemente.

### Factores Determinantes
El análisis de importancia de variables reveló que los **Cargos Mensuales**, el **Tipo de Contrato** y la **Antigüedad** son los tres pilares que definen la permanencia de un cliente en Telecom X.



---

## 6. Instrucciones de Ejecución

### Requisitos Previos
Es necesario tener instalado Python 3.x y las siguientes bibliotecas:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
