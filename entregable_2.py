

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import IsolationForest
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans, DBSCAN as SklearnDBSCAN
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree


#* Función para imprimir secciones con formato en la terminal
def printeo(x):
    """formatea informacion para mayor legibilidad

    Args:
        x (str, object): informacion o datos a imprimir
    """
    print(f"\n{'__'*60}\n{x}\n{'__'*60}\n")


# Cargar dataset
df = pd.read_csv('wine_quality_merged.csv')
df = df.sample(n=5150, random_state=41).reset_index(drop=True)
# Verificar la carga del dataset

printeo(f"Primeras filas del dataset: \n\n{df.head()}")


# CC: conversion categorica a numerica
df['type'] = df['type'].map({'red': 1, 'white': 0})
# Verificar la conversión

printeo(f"Dataset después de la conversión de categorica a numerica: \n\n{df.head()}")



# Función para preparar los datos según las variaciones de escalado, outliers y balanceo
def prepare_data(df, escalado, outliers, balanceo):
    """hace la conversion de la tabla en base a los parametros requeridos

    Args:
        df (table): dataframe con los datos originales
        escalado (bool): indicador de si se debe aplicar escalado
        outliers (bool): indicador de si se deben eliminar outliers
        balanceo (bool): indicador de si se debe balancear el dataset

    Returns:
        df_processed (table): dataframe procesado con las transformaciones aplicadas
        X (table): matriz de características después de las transformaciones
        y (series): vector objetivo después de las transformaciones
    """
    
    df_processed = df.copy()
    X = df_processed.drop('quality', axis=1)
    y = df_processed['quality']  # multiclase (3-9)

    if outliers == 'no':
        iso_forest = IsolationForest(contamination=0.05, random_state=41)
        outlier_mask = iso_forest.fit_predict(X) == 1
        X = X[outlier_mask]
        y = y[outlier_mask]

    if escalado == 'si':
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    if balanceo == 'si':
        smote = SMOTE(random_state=41, k_neighbors=4)
        X, y = smote.fit_resample(X, y)
        X = pd.DataFrame(X, columns=X.columns)

    return df_processed, X, y




# Generar tabla de variaciones con diferentes combinaciones de escalado, outliers y balanceo
variaciones = [
    {'escalado': 'no', 'outliers': 'no', 'balanceo': 'no'},
    {'escalado': 'no', 'outliers': 'no', 'balanceo': 'si'},
    {'escalado': 'no', 'outliers': 'si', 'balanceo': 'no'},
    {'escalado': 'no', 'outliers': 'si', 'balanceo': 'si'},
    {'escalado': 'si', 'outliers': 'no', 'balanceo': 'no'},
    {'escalado': 'si', 'outliers': 'no', 'balanceo': 'si'},
    {'escalado': 'si', 'outliers': 'si', 'balanceo': 'no'},
    {'escalado': 'si', 'outliers': 'si', 'balanceo': 'si'}
]



# Función para generar tabla resumen de variaciones en la terminal
def generar_tabla_variaciones(df, variaciones):
    """ genera una tabla en la terminal con el listado de las variaciones que se le va a hacer al csv original

    Args:
        df (table): dataframe con los datos originales
        variaciones (list): lista de diccionarios con los parámetros de cada variación

    Returns:
        filas: dataframe con el resumen de las variaciones
    """
    
    filas = []
    for i, params in enumerate(variaciones, 1):
        df_proc, X, y = prepare_data(df, **params)
        clases, counts = np.unique(y, return_counts=True)
        dist = dict(zip(clases, counts))
        filas.append({
            'Version': f'v{i}',
            'CC': 'si',
            'ED': params['escalado'],
            'Outliers': params['outliers'],
            'Balanceo': params['balanceo'],
            'Muestras': X.shape[0],
            'Features': X.shape[1],
            'Distribucion': dist
        })
    return pd.DataFrame(filas)

printeo(f"Tabla de variaciones aplicadas al dataset original:\n\n{generar_tabla_variaciones(df, variaciones)}")



# Guardar cada variacion en una variable para su posterior uso en el entrenamiento de modelos y comparación de resultados
v1 = prepare_data(df, **variaciones[0])  #caso 1(df_proc, X, y)
v2 = prepare_data(df, **variaciones[1]) #caso 2
v3 = prepare_data(df, **variaciones[2]) #caso 3
v4 = prepare_data(df, **variaciones[3]) #caso 4
v5 = prepare_data(df, **variaciones[4]) #caso 5
v6 = prepare_data(df, **variaciones[5]) #caso 6
v7 = prepare_data(df, **variaciones[6]) #caso 7
v8 = prepare_data(df, **variaciones[7]) #caso 8


# Verificacion de la primera variacion procesada
printeo(f"Primeras filas del DataFrame procesado de la variación v1:\n\n{v1[0].head()}")



# Función para calcular métricas de evaluación de modelos
def metricas(y_test, y_pred):
    """ calcula las metricas de evaluacion de los modelos

    Args:
        y_test (array-like): etiquetas reales del conjunto de prueba
        y_pred (array-like): etiquetas predichas por el modelo

    Returns:
        metricas: diccionario con las métricas de evaluación
    """
    
    
    metricas = {
        'accuracy': round(accuracy_score(y_test, y_pred), 2),
        'precision_weighted': round(precision_score(y_test, y_pred, average='weighted', zero_division=0),2),
        'recall_weighted': round(recall_score(y_test, y_pred, average='weighted', zero_division=0),2),
        'f1_weighted': round(f1_score(y_test, y_pred, average='weighted', zero_division=0),2),
        'precision_macro': round(precision_score(y_test, y_pred, average='macro', zero_division=0),2),
        'recall_macro': round(recall_score(y_test, y_pred, average='macro', zero_division=0),2),
        'f1_macro': round(f1_score(y_test, y_pred, average='macro', zero_division=0),2),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
    }
    return metricas





#*aprendizaje supervisado
def arboles(X,y):
    """ entrena un modelo de árbol de decisión utilizando el criterio de Gini y evalúa su desempeño

    Args: 
        X (pd.DataFrame): dataframe con las características de entrada para el modelo
        y (pd.Series): vector con las etiquetas de clase correspondientes a cada muestra en X

    Returns:
        metricas: diccionario con las métricas de evaluación del modelo
        importancia_vars: dataframe con la importancia de las variables en el modelo
        modelo_gini: modelo de árbol de decisión entrenado con el criterio de Gini
        Columna_x: lista con los nombres de las columnas utilizadas en el modelo
    """
    
    #Separacion de datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)

    #Crear y entrenar modelo
    modelo_gini = DecisionTreeClassifier(criterion='gini', max_depth=2, random_state=41)
    modelo_gini.fit(X_train, y_train)
    
    #Realizar predicciones
    y_pred = modelo_gini.predict(X_test)

    #Evaluacion del modelo
    metrica= metricas(y_test, y_pred)

    #Calcular importancia
    from collections import defaultdict

    modelos = {
        'Gini': modelo_gini,
    }

    for model_name, model in modelos.items():
        agrupadas = defaultdict(float)
        for feature, importance in zip(X_train.columns, model.feature_importances_):
            nombre_original = feature.split('_')[0]  # Asume que el prefijo antes del "_" es la variable original
            agrupadas[nombre_original] += importance

        # Paso 2: Convertir a DataFrame
        importancia_vars = pd.DataFrame({
            'Variable': list(agrupadas.keys()),
            'Importancia': list(agrupadas.values())
        }).sort_values(by='Importancia', ascending=False)
        
        
    return metrica, importancia_vars , modelo_gini, X_train.columns



def entrenar_knn(X, y, test_size=0.2, random_state=41):
    """ entrena un modelo de KNN utilizando la métrica euclidiana y evalúa su desempeño

    Args:
        X (pd.DataFrame): dataframe con las características de entrada para el modelo
        y (pd.Series): vector con las etiquetas de clase correspondientes a cada muestra en X
        test_size (float): proporción de datos a utilizar para testing.
        random_state (int): semilla para la generación de números aleatorios.

    Returns:
        knn: modelo de KNN entrenado
        metrica: diccionario con las métricas de evaluación del modelo
        accuracy: precisión del modelo en el conjunto de prueba
    """
    
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    metrica= metricas(y_test, y_pred)
    return knn, metrica, accuracy


def svm(X,y):
    """ entrena un modelo de SVM y evalúa su desempeño

    Args:
        X (pd.DataFrame): dataframe con las características de entrada para el modelo
        y (pd.Series): vector con las etiquetas de clase correspondientes a cada muestra en X

    Returns:
        metrica: diccionario con las métricas de evaluación del modelo
    """
    
    
    #Separacion de datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=41, stratify=y)
    
    #Creacion del modelo
    model = SVC(kernel='rbf', gamma="scale", C=1.0)  # C=1.0 (Equilibrado)
    model.fit(X_train, y_train)

    #Predicciones
    y_pred = model.predict(X_test)

    #Evaluacion del modelo
    metrica= metricas(y_test, y_pred)
    
    return metrica



def redes_neuronales(X, y):
    import tensorflow as tf
    from tensorflow.keras import models, layers,  regularizers # ← 1. Importar regularizers
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # ← 2. Importar callbacks

    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=41, stratify=y)

    # 3. Arquitectura con L2 y Dropout
    model = models.Sequential([
        layers.Dense(
            64, activation='relu',
            input_shape=(X_train.shape[1],),
            kernel_regularizer=regularizers.l2(0.001)  # ← penaliza pesos grandes
        ),
        layers.Dropout(0.2),   # ← apaga 20% neuronas al azar en cada epoch (evita overfitting)

        layers.Dense(
            32, activation='relu',
            kernel_regularizer=regularizers.l2(0.001)
        ),
        layers.Dropout(0.2),   # ← segundo Dropout

        layers.Dense(7, activation='softmax')  # 7 clases de calidad (3 a 9)
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # 4. Definir los callbacks
    early_stop = EarlyStopping(
        monitor='val_loss',   # observa la pérdida en validación
        patience=15,          # espera 15 epochs sin mejora antes de parar
        restore_best_weights=True  # recupera los mejores pesos encontrados
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',   # observa la pérdida en validación
        patience=5,           # espera 5 epochs sin mejora
        factor=0.5,           # reduce el learning rate a la mitad
        min_lr=1e-6           # límite mínimo del learning rate
    )

    #  5. Entrenar con los callbacks
    model.fit(
        X_train, y_train,
        epochs=100,            # ← ahora puede entrenarse hasta 100 epochs
        batch_size=32,
        validation_split=0.1,
        callbacks=[early_stop, reduce_lr],  # ← aquí se pasan los callbacks
        verbose=0
    )

    predicciones = model.predict(X_test, verbose=0)
    y_pred = np.argmax(predicciones, axis=1)
    metrica = metricas(y_test, y_pred)
    return metrica

#---

def grafico_comparacion_supervisada(arbolv1, knnv1, svmv1, red_neuronalv1, arbolv2, knnv2, svmv2, red_neuronalv2, arbolv3, knnv3, svmv3, red_neuronalv3, arbolv4, knnv4, svmv4, red_neuronalv4, 
                        arbolv5, knnv5, svmv5, red_neuronalv5, arbolv6, knnv6, svmv6, red_neuronalv6, arbolv7, knnv7, svmv7, red_neuronalv7, arbolv8, knnv8, svmv8, red_neuronalv8):
    
    """ genera una tabla comparativa en la terminal con las metricas de los 4 modelos (arbol, knn, svm, red_neuronal) para cada una de las 8 variaciones

    Args:
        arbolv1 a arbolv8 (list): lista con accuracy, precision_weighted, recall_weighted y f1_weighted para arbol en cada variacion
        knnv1 a knnv8 (list): lista con accuracy, precision_weighted, recall_weighted y f1_weighted para knn en cada variacion
        svmv1 a svmv8 (list): lista con accuracy, precision_weighted, recall_weighted y f1_weighted para svm en cada variacion
        red_neuronalv1 a red_neuronalv8 (list): lista con accuracy, precision_weighted, recall_weighted y f1_weighted para red_neuronal en cada variacion

    Returns:
        df: dataframe con la tabla comparativa de metricas organizada con MultiIndex
    """
        
    # Definir los algoritmos y métricas
    algoritmos = ['ÁRBOLES DE DECISIÓN', 'K VECINOS MAS CERCANOS (KNN)', 'MAQUINAS DE VECTORES DE SOPORTE (SVM)', 'REDES NEURONALES' ]
    metricas = ['ACCURACY', 'PRECISION', 'RECALL', 'F1-SCORE']

    # Crear MultiIndex para las columnas
    columnas_fijas = [('#', ''), ('Normalización', ''), ('OUTLIERS', ''), ('BALANCEO', '')]
    columnas_metricas = [(alg, met) for alg in algoritmos for met in metricas]

    multi_index = pd.MultiIndex.from_tuples(columnas_fijas + columnas_metricas)

    # Datos de las filas
    datos = [
        [1, 'CC(SI) y ED(NO)', 'NO', 'NO', arbolv1[0], arbolv1[1], arbolv1[2], arbolv1[3],
        knnv1[0], knnv1[1], knnv1[2], knnv1[3], svmv1[0], svmv1[1], svmv1[2], svmv1[3], red_neuronalv1[0], red_neuronalv1[1], red_neuronalv1[2], red_neuronalv1[3]],
        
        [2, 'CC(SI) y ED(NO)', 'NO', 'SI', arbolv2[0], arbolv2[1], arbolv2[2], arbolv2[3], 
        knnv2[0], knnv2[1], knnv2[2], knnv2[3], svmv2[0], svmv2[1], svmv2[2], svmv2[3], red_neuronalv2[0], red_neuronalv2[1], red_neuronalv2[2], red_neuronalv2[3]],
        
        [3, 'CC(SI) y ED(NO)', 'SI', 'NO', arbolv3[0], arbolv3[1], arbolv3[2], arbolv3[3], 
        knnv3[0], knnv3[1], knnv3[2], knnv3[3],svmv3[0], svmv3[1], svmv3[2], svmv3[3], red_neuronalv3[0], red_neuronalv3[1], red_neuronalv3[2], red_neuronalv3[3]],
        
        [4, 'CC(SI) y ED(NO)', 'SI', 'SI', arbolv4[0], arbolv4[1], arbolv4[2], arbolv4[3], 
        knnv4[0], knnv4[1], knnv4[2], knnv4[3], svmv4[0], svmv4[1], svmv4[2], svmv4[3], red_neuronalv4[0], red_neuronalv4[1], red_neuronalv4[2], red_neuronalv4[3]],
        
        [5, 'CC(SI) y ED(SI)', 'NO', 'NO', arbolv5[0], arbolv5[1], arbolv5[2], arbolv5[3], 
        knnv5[0], knnv5[1], knnv5[2], knnv5[3], svmv5[0], svmv5[1], svmv5[2], svmv5[3], red_neuronalv5[0], red_neuronalv5[1], red_neuronalv5[2], red_neuronalv5[3]],
        
        [6, 'CC(SI) y ED(SI)', 'NO', 'SI', arbolv6[0], arbolv6[1], arbolv6[2], arbolv6[3], 
        knnv6[0], knnv6[1], knnv6[2], knnv6[3], svmv6[0], svmv6[1], svmv6[2], svmv6[3], red_neuronalv6[0], red_neuronalv6[1], red_neuronalv6[2], red_neuronalv6[3]],
        
        [7, 'CC(SI) y ED(SI)', 'SI', 'NO', arbolv7[0], arbolv7[1], arbolv7[2], arbolv7[3], 
        knnv7[0], knnv7[1], knnv7[2], knnv7[3], svmv7[0], svmv7[1], svmv7[2], svmv7[3], red_neuronalv7[0], red_neuronalv7[1], red_neuronalv7[2], red_neuronalv7[3]],
        
        [8, 'CC(SI) y ED(SI)', 'SI', 'SI', arbolv8[0], arbolv8[1], arbolv8[2], arbolv8[3], 
        knnv8[0], knnv8[1], knnv8[2], knnv8[3], svmv8[0], svmv8[1], svmv8[2], svmv8[3], red_neuronalv8[0], red_neuronalv8[1], red_neuronalv8[2], red_neuronalv8[3]],
    ]

    df = pd.DataFrame(datos, columns=multi_index)
    
    return df


def graficar_tabla_comparacion(tabla):
    """Muestra una tabla comparativa de métricas de modelos en una figura de Matplotlib.

    Args:
        tabla (pd.DataFrame): dataframe con métricas organizadas en MultiIndex.

    Returns:
        None: renderiza la tabla en un gráfico de Matplotlib.
    """
    fig, ax = plt.subplots(figsize=(20, 6))
    ax.axis('off')
    ax.axis('tight')
    ax.table(cellText=tabla.values,
            colLabels=tabla.columns.get_level_values(1) if isinstance(tabla.columns, pd.MultiIndex) else tabla.columns,
            rowLabels=tabla.index,
            cellLoc='center',
            loc='center')
    plt.title("Comparación de modelos por variación", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

def graficar_f1_maximo_supervisado(arboles, knns, svms, redes):
    """Genera un gráfico de barras con el F1 máximo por tipo de modelo.

    Args:
        arboles (list[list[float]]): métricas de las variaciones del modelo de árbol.
        knns (list[list[float]]): métricas de las variaciones del modelo KNN.
        svms (list[list[float]]): métricas de las variaciones del modelo SVM.
        redes (list[list[float]]): métricas de las variaciones del modelo de redes neuronales.

    Returns:
        None: muestra el gráfico de barras con los valores máximos de F1 para cada algoritmo.
    """
    algoritmos = ['ÁRBOLES\nDE DECISIÓN', 'KNN', 'SVM', 'REDES\nNEURONALES']
    f1_maximos = [
        max(a[3] for a in arboles),
        max(k[3] for k in knns),
        max(s[3] for s in svms),
        max(r[3] for r in redes),
    ]

    colores = ['#2E7D32', '#1565C0', '#E65100', '#6A1B9A']

    plt.figure(figsize=(8, 5))
    bars = plt.bar(algoritmos, f1_maximos, color=colores, edgecolor='black', linewidth=1.2)

    for bar, valor in zip(bars, f1_maximos):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{valor:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.ylim(0, 1.1)
    plt.ylabel('F1-Score Máximo', fontsize=12)
    plt.title('F1-Score máximo por algoritmo (supervisado)', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

def graficar_f1_medio_por_normalizacion(arboles, knns, svms, redes):
    """Grafica el F1 medio por algoritmo comparando datos con y sin normalización.

    Args:
        arboles (list[list[float]]): métricas de árbol de decisión para cada variación.
        knns (list[list[float]]): métricas de KNN para cada variación.
        svms (list[list[float]]): métricas de SVM para cada variación.
        redes (list[list[float]]): métricas de redes neuronales para cada variación.

    Returns:
        None: muestra un gráfico de barras de F1 medio para ED=NO y ED=SI.
    """
    import numpy as np

    def media_f1(modelos):
        """Calcula el F1 medio de una lista de métricas de modelo.

        Args:
            modelos (list[list[float]]): lista de métricas de modelo, donde el índice 3 es F1.

        Returns:
            float: valor medio de F1 para los modelos proporcionados.
        """
        return np.mean([m[3] for m in modelos])

    f1_ed_no = [
        media_f1(arboles[:4]),
        media_f1(knns[:4]),
        media_f1(svms[:4]),
        media_f1(redes[:4]),
    ]
    f1_ed_si = [
        media_f1(arboles[4:]),
        media_f1(knns[4:]),
        media_f1(svms[4:]),
        media_f1(redes[4:]),
    ]

    categorias = ['ÁRBOLES', 'KNN', 'SVM', 'REDES\nNEURONALES']
    x = np.arange(len(categorias))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, f1_ed_no, width, label='ED=NO', color='#42A5F5', edgecolor='black')
    bars2 = ax.bar(x + width/2, f1_ed_si, width, label='ED=SI', color='#EF5350', edgecolor='black')

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(categorias, fontsize=11)
    ax.set_ylabel('F1-Score medio', fontsize=12)
    ax.set_title('F1-Score medio por normalización y algoritmo', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

def graficar_f1_medio_por_outliers(arboles, knns, svms, redes):
    """Grafica el F1 medio por algoritmo comparando datos con y sin outliers.

    Args:
        arboles (list[list[float]]): métricas de árbol de decisión para cada variación.
        knns (list[list[float]]): métricas de KNN para cada variación.
        svms (list[list[float]]): métricas de SVM para cada variación.
        redes (list[list[float]]): métricas de redes neuronales para cada variación.

    Returns:
        None: muestra un gráfico de barras de F1 medio para outliers=NO y outliers=SI.
    """
    import numpy as np

    def media_f1(modelos):
        """Calcula el F1 medio de una lista de métricas de modelo.

        Args:
            modelos (list[list[float]]): lista de métricas de modelo, donde el índice 3 es F1.

        Returns:
            float: valor medio de F1 para los modelos proporcionados.
        """
        return np.mean([m[3] for m in modelos])

    f1_out_no = [
        media_f1([arboles[i] for i in [0,1,4,5]]),
        media_f1([knns[i] for i in [0,1,4,5]]),
        media_f1([svms[i] for i in [0,1,4,5]]),
        media_f1([redes[i] for i in [0,1,4,5]]),
    ]
    f1_out_si = [
        media_f1([arboles[i] for i in [2,3,6,7]]),
        media_f1([knns[i] for i in [2,3,6,7]]),
        media_f1([svms[i] for i in [2,3,6,7]]),
        media_f1([redes[i] for i in [2,3,6,7]]),
    ]

    categorias = ['ÁRBOLES', 'KNN', 'SVM', 'REDES\nNEURONALES']
    x = np.arange(len(categorias))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, f1_out_no, width, label='Outliers=NO', color='#66BB6A', edgecolor='black')
    bars2 = ax.bar(x + width/2, f1_out_si, width, label='Outliers=SI', color='#FFA726', edgecolor='black')

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(categorias, fontsize=11)
    ax.set_ylabel('F1-Score medio', fontsize=12)
    ax.set_title('F1-Score medio por manejo de outliers y algoritmo', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

def graficar_f1_medio_por_balanceo(arboles, knns, svms, redes):
    """Grafica el F1 medio por algoritmo comparando datos con y sin balanceo.

    Args:
        arboles (list[list[float]]): métricas de árbol de decisión para cada variación.
        knns (list[list[float]]): métricas de KNN para cada variación.
        svms (list[list[float]]): métricas de SVM para cada variación.
        redes (list[list[float]]): métricas de redes neuronales para cada variación.

    Returns:
        None: muestra un gráfico de barras de F1 medio para balanceo=NO y balanceo=SI.
    """
    import numpy as np

    def media_f1(modelos):
        """Calcula el F1 medio de una lista de métricas de modelo.

        Args:
            modelos (list[list[float]]): lista de métricas de modelo, donde el índice 3 es F1.

        Returns:
            float: valor medio de F1 para los modelos proporcionados.
        """
        return np.mean([m[3] for m in modelos])

    f1_bal_no = [
        media_f1([arboles[i] for i in [0,2,4,6]]),
        media_f1([knns[i] for i in [0,2,4,6]]),
        media_f1([svms[i] for i in [0,2,4,6]]),
        media_f1([redes[i] for i in [0,2,4,6]]),
    ]
    f1_bal_si = [
        media_f1([arboles[i] for i in [1,3,5,7]]),
        media_f1([knns[i] for i in [1,3,5,7]]),
        media_f1([svms[i] for i in [1,3,5,7]]),
        media_f1([redes[i] for i in [1,3,5,7]]),
    ]

    categorias = ['ÁRBOLES', 'KNN', 'SVM', 'REDES\nNEURONALES']
    x = np.arange(len(categorias))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, f1_bal_no, width, label='Balanceo=NO', color='#AB47BC', edgecolor='black')
    bars2 = ax.bar(x + width/2, f1_bal_si, width, label='Balanceo=SI', color='#26C6DA', edgecolor='black')

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(categorias, fontsize=11)
    ax.set_ylabel('F1-Score medio', fontsize=12)
    ax.set_title('F1-Score medio por balanceo y algoritmo', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

def graficar_roc_pr_v8(X, y):
    """Genera curvas ROC y Precision-Recall para cuatro modelos sobre una variación.

    Args:
        X (pd.DataFrame or np.ndarray): características de entrada.
        y (pd.Series or np.ndarray): etiquetas de clase.

    Returns:
        None: muestra las curvas ROC y PR para los modelos evaluados.
    """
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.svm import SVC
    import tensorflow as tf
    from tensorflow.keras import models, layers
    from sklearn.preprocessing import LabelEncoder

    encoder = LabelEncoder()
    y_enc = encoder.fit_transform(y)
    classes = np.unique(y_enc)
    n_classes = len(classes)

    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.30, random_state=41, stratify=y_enc)
    y_test_bin = label_binarize(y_test, classes=classes)

    modelos = {}

    # Árbol
    from sklearn.tree import DecisionTreeClassifier
    arbol = DecisionTreeClassifier(criterion='gini', max_depth=2, random_state=41)
    arbol.fit(X_train, y_train)
    modelos['ÁRBOL'] = arbol.predict_proba(X_test)

    # KNN
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
    knn.fit(X_train, y_train)
    modelos['KNN'] = knn.predict_proba(X_test)

    # SVM (con probability=True)
    svm = SVC(kernel='rbf', gamma="scale", C=1.0, probability=True, random_state=41)
    svm.fit(X_train, y_train)
    modelos['SVM'] = svm.predict_proba(X_test)

    # Red Neuronal
    red = models.Sequential([
        layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
        layers.Dense(16, activation='relu'),
        layers.Dense(n_classes, activation='softmax')
    ])
    red.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    red.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1, verbose=0)
    modelos['RED\nNEURONAL'] = red.predict(X_test, verbose=0)

    colores = {'ÁRBOL': '#2E7D32', 'KNN': '#1565C0', 'SVM': '#E65100', 'RED\nNEURONAL': '#6A1B9A'}

    # ROC
    plt.figure(figsize=(8, 7))
    for nombre, prob in modelos.items():
        fpr, tpr, _ = roc_curve(y_test_bin.ravel(), prob.ravel())
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colores[nombre], lw=2, label=f'{nombre} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos', fontsize=12)
    plt.ylabel('Tasa de Verdaderos Positivos', fontsize=12)
    plt.title('Curvas ROC - Variación 8', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # PR
    plt.figure(figsize=(8, 7))
    for nombre, prob in modelos.items():
        precision, recall, _ = precision_recall_curve(y_test_bin.ravel(), prob.ravel())
        ap = average_precision_score(y_test_bin, prob, average='micro')
        plt.plot(recall, precision, color=colores[nombre], lw=2, label=f'{nombre} (AP = {ap:.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Curvas Precision-Recall - Variación 8', fontsize=14, fontweight='bold')
    plt.legend(loc='lower left', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()




# calcular metricas de cada modelo para cada variacion y generar la tabla comparativa
def variacion_supervisada(variacion):
    """ ejecuta los 3 modelos supervisados (arbol, knn, svm) sobre una variacion y retorna sus metricas y objetos entrenados

    Args:
        variacion (tuple): tupla de 3 elementos (df_procesado, X, y) generada por prepare_data

    Returns:
        arbol (list): metricas del arbol [accuracy, precision_weighted, recall_weighted, f1_weighted]
        knn (list): metricas del knn [accuracy, precision_weighted, recall_weighted, f1_weighted]
        svm_metrics (list): metricas del svm [accuracy, precision_weighted, recall_weighted, f1_weighted]
        modelo: modelo de arbol de decision entrenado
        Columna_x: nombres de las columnas usadas en el modelo
        X: dataframe de caracteristicas
        y: serie de etiquetas
        knn_model: modelo knn entrenado
    """
    _, X, y = variacion
    
    
    
    #arboles 
    metrica_arbol, importancia_vars, modelo, Columna_x = arboles(X,y)
    print(f"Arboles metricas: {metrica_arbol}\n")
    print(f"Importancia de variables:\n{importancia_vars}\n")
    #guardar metricas en una variable para la tabla comparativa
    arbol = [metrica_arbol['accuracy'],metrica_arbol["precision_weighted"], 
                metrica_arbol['recall_weighted'], metrica_arbol['f1_weighted']]
    
    
    #knn
    knn_model, metrica_knn, accuracy = entrenar_knn(X, y)
    print(f"KNN Accuracy: {accuracy:.4f}\n")
    #guardar metricas en una variable para la tabla comparativa
    knn = [metrica_knn['accuracy'],metrica_knn["precision_weighted"], 
            metrica_knn['recall_weighted'], metrica_knn['f1_weighted']]
    
    #SVM
    print("calculando metricas SVM...")
    metrica_svm = svm(X,y)
    #TODO metrica_svm, svm_model, X_test_svm, y_test_svm = svm(X,y)

    print(f"Metricas svm: {metrica_svm}\n")
    svm_metrics = [metrica_svm['accuracy'],metrica_svm["precision_weighted"], 
            metrica_svm['recall_weighted'], metrica_svm['f1_weighted']]
    
    
    #redes neuronales
    print("calculando Red neuronal...")
    metrica_red_neuronal = redes_neuronales(X,y)
    print(f"Metricas redes neuronales: {metrica_red_neuronal}\n")
    red_neuronal_metrics = [metrica_red_neuronal['accuracy'],metrica_red_neuronal["precision_weighted"], 
            metrica_red_neuronal['recall_weighted'], metrica_red_neuronal['f1_weighted']]
    
    
    return arbol, knn, svm_metrics, modelo, Columna_x, X, y, knn_model, red_neuronal_metrics




# calcular metricas de cada modelo para cada variacion y generar la tabla comparativa
def variacion_no_supervisada(variacion):
    """ ejecuta KMeans y DBSCAN sobre una variacion y retorna sus resultados

    Args:
        variacion (tuple): tupla de 3 elementos (df_procesado, X, y) generada por prepare_data

    Returns:
        kmeans_report (dict): reporte completo de train_kmeans_models
        kmeans_elbow: modelo KMeans entrenado con k optimo por el codo
        kmeans_silhouette: modelo KMeans entrenado con k optimo por la silueta
        elbow_k (int): k optimo segun el codo
        silhouette_k (int): k optimo segun la silueta
        dbscan_report (dict): reporte completo de train_dbscan_model
        dbscan_model: modelo DBSCAN entrenado
        dbscan_eps (float): valor de eps utilizado
        dbscan_labels: etiquetas de cluster asignadas por DBSCAN
    """
    _, X, _ = variacion

    
    #KMeans
    print("calculando KMeans...")
    kmeans_report = train_kmeans_models(X)
    elbow_k = kmeans_report['selected_k_elbow']
    silhouette_k = kmeans_report['selected_k_silhouette']
    kmeans_elbow = kmeans_report['strategies']['elbow']['model']
    kmeans_silhouette = kmeans_report['strategies']['silhouette']['model']
    print(f"K optimo (elbow): {elbow_k}")
    print(f"K optimo (silueta): {silhouette_k}")

    #DBSCAN
    print("calculando DBSCAN...")
    dbscan_report = train_dbscan_model(X)
    dbscan_model = dbscan_report['model']
    dbscan_eps = dbscan_report['eps']
    dbscan_labels = dbscan_report['labels']
    print(f"DBSCAN - eps: {dbscan_eps}, clusters: {dbscan_report['cluster_count']}, ruido: {dbscan_report['noise_count']}")

    
    
    return (kmeans_report, kmeans_elbow, kmeans_silhouette, elbow_k, silhouette_k,
            dbscan_report, dbscan_model, dbscan_eps, dbscan_labels)



from sklearn.decomposition import PCA

def graficar_arbol(modelo, feature_names):
        """ 
        genera una visualizacion grafica del arbol de decision entrenado

        Args:
            modelo: modelo de arbol de decision entrenado (DecisionTreeClassifier)
            feature_names: nombres de las columnas de caracteristicas utilizadas

        Returns:
            None: muestra la grafica con plt.show()
        """
        #feature_names = X_train.columns

        plt.figure(figsize=(15, 10))  # Tamaño ajustado
        plot_tree(
            modelo, ##Nombre con el que guardo el modelo
            filled=True,
            feature_names=feature_names,
            class_names=modelo.classes_.astype(str), ##Nombre con el que guardo el modelo
            rounded=True,
            fontsize=8
        )
        plt.title("Árbol de Decisión", fontsize=12)
        plt.tight_layout()
        plt.show()


def graficar_knn(X, y, knn, titulo="KNN - Clasificacion"):
    """ genera una visualizacion de la clasificacion KNN usando PCA para reducir a 2 dimensiones

    Args:
        X: dataframe de caracteristicas de entrada
        y: serie con las etiquetas de clase reales
        knn: modelo KNN entrenado (KNeighborsClassifier)
        titulo (str): titulo opcional para la grafica (default "KNN - Clasificacion")

    Returns:
        None: muestra la grafica con plt.show()
    """

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    y_pred = knn.predict(X)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10', alpha=0.6, edgecolors='k', s=20)
    axes[0].set_title("Clases Reales")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")

    axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=y_pred, cmap='tab10', alpha=0.6, edgecolors='k', s=20)
    axes[1].set_title("Clases Predichas (KNN)")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")

    plt.suptitle(titulo)
    plt.tight_layout()
    plt.show()

def graficar_svm(X, y, class_labels):
    """Visualiza la frontera de decisión de un SVM sobre los datos reducidos por PCA.

    Args:
        X (pd.DataFrame or np.ndarray): características de entrada.
        y (pd.Series or np.ndarray): etiquetas reales de clases.
        class_labels (list|np.ndarray): etiquetas únicas de clase para la leyenda.

    Returns:
        None: muestra un gráfico de contornos con las regiones de clasificación.
    """
    # Reducir a 2 dimensiones
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # Modelo SVM para visualización
    model_vis = SVC(kernel='rbf', gamma='scale', C=1.0)
    model_vis.fit(X_pca, y)

    # Crear malla
    h = 0.02

    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1

    """xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h), # h=0.02 → puede crear millones de puntos
        np.arange(y_min, y_max, h)
    )"""
    
    xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 200),  # siempre 200 puntos
    np.linspace(y_min, y_max, 200)   # siempre 200 puntos → 40,000 total (rápido)
    )


    # Predicción sobre la malla
    Z = model_vis.predict(
        np.c_[xx.ravel(), yy.ravel()]
    )

    Z = Z.reshape(xx.shape)

    # Gráfico
    plt.figure(figsize=(10,8))

    plt.contourf(
        xx,
        yy,
        Z,
        alpha=0.3,
        cmap=plt.cm.RdYlBu
    )

    scatter = plt.scatter(
        X_pca[:,0],
        X_pca[:,1],
        c=y,
        cmap=plt.cm.RdYlBu,
        edgecolor='black'
    )

    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.title('Fronteras de decisión SVM (PCA)')

    plt.legend(
        handles=scatter.legend_elements()[0],
        labels=[str(c) for c in class_labels],
        title="Quality"
    )

    plt.show()
    
# funcion para graficar el arbol de decision y la clasificacion knn de una variacion dada
def graficar_funciones_supervisadas(knn,modelo, Columna_x, X, y):
    """ grafica el arbol de decision y la clasificacion KNN de una variacion

    Args:
        knn: modelo KNN entrenado (KNeighborsClassifier)
        modelo: modelo de arbol de decision entrenado (DecisionTreeClassifier)
        Columna_x: nombres de las columnas de caracteristicas utilizadas
        X: dataframe de caracteristicas de entrada
        y: serie con las etiquetas de clase reales

    Returns:
        None: muestra las graficas con plt.show()
    """
    graficar_arbol(modelo, Columna_x)
    graficar_knn(X, y, knn)
    graficar_svm(X, y, np.unique(y))







#* aprendizaje no supervisado

from sklearn.metrics import silhouette_score  
try:
    from kneed import KneeLocator
except ImportError:
    KneeLocator = None


# funcion para calcular la inercia de KMeans para diferentes valores de k 
def _compute_kmeans_inertia(X, k_values, random_state=41):
    """ calcula la inercia de KMeans para diferentes valores de k

    Args:
        X: dataframe o array con las caracteristicas de entrada
        k_values (list): lista de valores de k a evaluar
        random_state (int): semilla para reproducibilidad (default 41)

    Returns:
        inertia (list): lista con los valores de inercia para cada k
    """
    inertia = []
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        kmeans.fit(X)
        inertia.append(kmeans.inertia_)
    return inertia

# funcion para seleccionar el valor optimo de k usando el metodo del codo (elbow) 
def _select_k_by_elbow(k_values, inertia, fallback_k):
    """ selecciona el valor optimo de k usando el metodo del codo (elbow) con KneeLocator

    Args:
        k_values (list): lista de valores de k evaluados
        inertia (list): lista de valores de inercia correspondientes a cada k
        fallback_k (int): valor de k por defecto si no se detecta un codo claro

    Returns:
        int: valor optimo de k detectado, o fallback_k si no se encuentra
    """
    if KneeLocator is None or len(k_values) < 2:
        return fallback_k

    locator = KneeLocator(k_values, inertia, curve='convex', direction='decreasing')
    return locator.elbow if locator.elbow is not None else fallback_k


# funcion para calcular el puntaje de la silueta para diferentes valores de k y seleccionar el optimo
def _compute_silhouette_scores(X, k_values, random_state=41):
    """ calcula el puntaje de silueta para diferentes valores de k y selecciona el optimo

    Args:
        X: dataframe o array con las caracteristicas de entrada
        k_values (list): lista de valores de k a evaluar
        random_state (int): semilla para reproducibilidad (default 41)

    Returns:
        selected_k (int): valor de k con mayor puntaje de silueta
        best_score (float): puntaje de silueta del k seleccionado
        silhouette_scores (list): lista con todos los puntajes de silueta calculados
    """
    silhouette_scores = []

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(X)
        silhouette_scores.append(silhouette_score(X, labels))

    if silhouette_scores:
        best_index = int(np.argmax(silhouette_scores))
        selected_k = k_values[best_index]
        selected_score = silhouette_scores[best_index]
    else:
        selected_k = None
        selected_score = None

    return silhouette_scores, selected_k, selected_score


# funcion para entrenar modelos KMeans usando los metodos del codo y silueta 
def train_kmeans_models(X, random_state=41, k_range=range(1, 11), silhouette_k_range=range(2, 11)):
    """ entrena modelos KMeans usando los metodos del codo y silueta para seleccionar k optimo

    Args:
        X: dataframe o array con las caracteristicas de entrada
        random_state (int): semilla para reproducibilidad (default 41)
        k_range (range): rango de valores de k para el metodo del codo (default range(1, 11))
        silhouette_k_range (range): rango de valores de k para el metodo de silueta (default range(2, 11))

    Returns:
        dict: diccionario con k_values, inertia, silhouette_scores, selected_k_elbow,
              selected_k_silhouette, y strategies con modelos y etiquetas entrenados
              para los metodos elbow y silhouette
    """
    X_train = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
    k_values = list(k_range)
    silhouette_values = list(silhouette_k_range)

    inertia = _compute_kmeans_inertia(X_train, k_values, random_state=random_state)
    fallback_k = silhouette_values[0] if silhouette_values else (k_values[0] if k_values else 2)
    elbow_k = _select_k_by_elbow(k_values, inertia, fallback_k)
    silhouette_scores, silhouette_k, silhouette_best_score = _compute_silhouette_scores(
        X_train,
        silhouette_values,
        random_state=random_state,
    )

    elbow_model = None
    elbow_labels = None
    elbow_silhouette = None
    if elbow_k is not None:
        elbow_model = KMeans(n_clusters=elbow_k, random_state=random_state, n_init=10)
        elbow_labels = elbow_model.fit_predict(X_train)
        if len(np.unique(elbow_labels)) > 1:
            elbow_silhouette = silhouette_score(X_train, elbow_labels)

    silhouette_model = None
    silhouette_labels = None
    silhouette_model_score = None
    if silhouette_k is not None:
        silhouette_model = KMeans(n_clusters=silhouette_k, random_state=random_state, n_init=10)
        silhouette_labels = silhouette_model.fit_predict(X_train)
        if len(np.unique(silhouette_labels)) > 1:
            silhouette_model_score = silhouette_score(X_train, silhouette_labels)

    return {
        # Valores evaluados para la graficacion de los valores del codo y silueta
        'k_values': k_values,
        'silhouette_k_values': silhouette_values,
        'inertia': inertia,
        'silhouette_scores': silhouette_scores,
        'selected_k_elbow': elbow_k,
        'selected_k_silhouette': silhouette_k,
        'selected_silhouette_score': silhouette_best_score,
        'strategies': {
            # Modelos y resultados para cada estrategia de selección de k
            'elbow': {
                'selected_k': elbow_k,
                'inertia': elbow_model.inertia_ if elbow_model is not None else None,
                'silhouette_score': elbow_silhouette,
                'labels': elbow_labels,
                'model': elbow_model,
            },
            # Modelos y resultados para la estrategia de selección basada en silueta
            'silhouette': {
                'selected_k': silhouette_k,
                'inertia': silhouette_model.inertia_ if silhouette_model is not None else None,
                'silhouette_score': silhouette_model_score,
                'labels': silhouette_labels,
                'model': silhouette_model,
            },
        },
    }





def _as_numpy_matrix(X):
    """ convierte un dataframe o array a un numpy array

    Args:
        X: dataframe de pandas o array-like a convertir

    Returns:
        np.ndarray: representacion como numpy array
    """
    if isinstance(X, pd.DataFrame):
        return X.to_numpy()
    return np.asarray(X)


def _safe_dbscan_silhouette_score(X, labels, noise_label=-1):
    """ calcula el silhouette score para DBSCAN excluyendo puntos etiquetados como ruido

    Args:
        X: dataframe o array con las caracteristicas de entrada
        labels: array con las etiquetas de cluster asignadas por DBSCAN
        noise_label (int): valor de la etiqueta que representa ruido (default -1)

    Returns:
        float or None: silhouette score sobre puntos no ruido, o None si hay menos de 2 clusters
    """
    X_matrix = _as_numpy_matrix(X)
    labels = np.asarray(labels)
    usable_mask = labels != noise_label
    usable_labels = labels[usable_mask]

    if usable_labels.size == 0:
        return None

    unique_labels = np.unique(usable_labels)
    if unique_labels.size < 2:
        return None

    X_usable = X_matrix[usable_mask]
    if X_usable.shape[0] <= unique_labels.size:
        return None

    try:
        return silhouette_score(X_usable, usable_labels)
    except ValueError:
        return None


def _estimate_dbscan_eps(X, min_samples, fallback_percentile=90):
    """ estima el valor optimo de eps para DBSCAN usando el metodo de la distancia k mas cercana

    Args:
        X: dataframe o array con las caracteristicas de entrada
        min_samples (int): valor de min_samples que se usara en DBSCAN
        fallback_percentile (int): percentil para calcular eps por defecto si no se detecta codo (default 90)

    Returns:
        eps (float): valor de eps estimado
        k_distances (np.ndarray): distancias ordenadas al k-vecino mas cercano
    """
    X_matrix = _as_numpy_matrix(X)
    n_samples = X_matrix.shape[0]

    if n_samples == 0:
        return 0.5, np.array([])

    n_neighbors = max(1, min(int(min_samples), n_samples))
    neighbors = NearestNeighbors(n_neighbors=n_neighbors)
    neighbors.fit(X_matrix)
    distances, _ = neighbors.kneighbors(X_matrix)
    k_distances = np.sort(distances[:, -1])

    eps = None
    if KneeLocator is not None and len(k_distances) >= 3:
        x_values = np.arange(1, len(k_distances) + 1)
        locator = KneeLocator(x_values, k_distances, curve='convex', direction='increasing')
        if locator.knee is not None:
            knee_index = int(locator.knee) - 1
            if 0 <= knee_index < len(k_distances):
                eps = float(k_distances[knee_index])

    if eps is None:
        eps = float(np.percentile(k_distances, fallback_percentile))

    if not np.isfinite(eps) or eps <= 0:
        positive_distances = k_distances[k_distances > 0]
        eps = float(np.max(positive_distances)) if positive_distances.size else 1e-6

    return max(eps, 1e-6), k_distances


def _build_dbscan_eps_candidates(estimated_eps, k_distances, fallback_percentile=90):
    """ genera candidatos de eps para DBSCAN a partir del estimado y percentiles de distancias k

    Args:
        estimated_eps (float): valor de eps estimado previamente
        k_distances (np.ndarray): distancias ordenadas al k-vecino mas cercano
        fallback_percentile (int): percentil usado como referencia (default 90)

    Returns:
        dict: diccionario con candidatos de eps, donde cada entrada tiene 'eps' y 'sources'
    """
    candidate_map = {}

    def _add_candidate(value, source):
        """Agrega un candidato de eps a la lista de candidatos de DBSCAN.

        Args:
            value (float): valor de eps a evaluar.
            source (str): origen o etiqueta de este candidato.

        Returns:
            None: modifica candidate_map en el cierre exterior.
        """
        if value is None:
            return
        if not np.isfinite(value) or value <= 0:
            return

        key = round(float(value), 6)
        entry = candidate_map.setdefault(key, {'eps': float(value), 'sources': []})
        if source not in entry['sources']:
            entry['sources'].append(source)

    _add_candidate(estimated_eps, 'estimado')

    if estimated_eps is not None and np.isfinite(estimated_eps) and estimated_eps > 0:
        for factor in (0.75, 0.9, 1.0, 1.1, 1.25):
            _add_candidate(estimated_eps * factor, f'multiplicador x{factor}')

    k_distances = np.asarray(k_distances)
    if k_distances.size:
        for percentile in (80, 85, 90, 95):
            _add_candidate(np.percentile(k_distances, percentile), f'percentil {percentile}')

    if not candidate_map and k_distances.size:
        _add_candidate(np.percentile(k_distances, fallback_percentile), f'fallback percentil {fallback_percentile}')

    return sorted(candidate_map.values(), key=lambda item: item['eps'])


def _evaluate_dbscan_candidate(X, min_samples, candidate):
    """Evalúa un candidato de configuración para DBSCAN.

    Args:
        X: dataframe o array con las características de entrada.
        min_samples (int): valor de min_samples para DBSCAN.
        candidate (dict): candidato con claves 'eps' y 'sources'.

    Returns:
        dict: resultados del candidato incluyendo modelo, etiquetas, conteo de clusters, ruido y puntuación de silueta.
    """
    X_matrix = _as_numpy_matrix(X)
    eps = float(candidate['eps'])
    model = SklearnDBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X_matrix)

    cluster_labels = sorted(int(label) for label in np.unique(labels) if label != -1)
    cluster_count = len(cluster_labels)
    noise_count = int(np.sum(labels == -1))
    silhouette = _safe_dbscan_silhouette_score(X_matrix, labels)
    valid_for_selection = silhouette is not None and cluster_count >= 2 and noise_count < len(labels)

    return {
        **candidate,
        'model': model,
        'labels': labels,
        'cluster_labels': cluster_labels,
        'cluster_count': cluster_count,
        'noise_count': noise_count,
        'silhouette_score': silhouette,
        'valid_for_selection': valid_for_selection,
        'status': 'válido' if valid_for_selection else ('sin_silhouette' if cluster_count >= 1 else 'colapso_total'),
    }


def _select_best_dbscan_candidate(candidate_results, fallback_result):
    """Selecciona el mejor candidato DBSCAN según métrica de silueta y calidad de cluster.

    Args:
        candidate_results (list[dict]): resultados de evaluación de candidatos de eps.
        fallback_result (dict): resultado a usar si no hay candidatos válidos.

    Returns:
        tuple: candidato seleccionado, motivo de selección, si se usó fallback, y lista de candidatos válidos.
    """
    valid_candidates = [candidate for candidate in candidate_results if candidate['valid_for_selection']]

    if valid_candidates:
        selected = max(
            valid_candidates,
            key=lambda candidate: (
                candidate['silhouette_score'],
                candidate['cluster_count'],
                -candidate['noise_count'],
                -candidate['eps'],
            ),
        )
        selection_reason = 'mejor silhouette válida entre los candidatos'
        used_fallback = False
    else:
        selected = fallback_result if fallback_result is not None else (candidate_results[0] if candidate_results else None)
        selection_reason = 'fallback al eps estimado original'
        used_fallback = True

    return selected, selection_reason, used_fallback, valid_candidates


def train_dbscan_model(X, min_samples=None):
    """Entrena y selecciona el mejor modelo DBSCAN para un conjunto de datos.

    Args:
        X: dataframe o array con las características de entrada.
        min_samples (int, optional): valor inicial de min_samples para DBSCAN.

    Returns:
        dict: informe con modelo seleccionado, eps, etiquetas, conteo de clusters y métricas de silueta.
    """
    X_matrix = _as_numpy_matrix(X)
    if X_matrix.size == 0:
        return {
            'model': None,
            'labels': np.array([]),
            'eps': None,
            'estimated_eps': None,
            'min_samples': None,
            'cluster_labels': [],
            'cluster_count': 0,
            'noise_count': 0,
            'silhouette_score': None,
            'k_distances': np.array([]),
            'eps_candidates': [],
            'selection_reason': 'sin datos',
            'selected_from_fallback': True,
        }

    n_samples, feature_count = X_matrix.shape
    if min_samples is None:
        min_samples = max(3, feature_count + 1)
    min_samples = int(min_samples)
    min_samples = min(min_samples, n_samples)
    min_samples = max(2, min_samples) if n_samples >= 2 else n_samples

    estimated_eps, k_distances = _estimate_dbscan_eps(X_matrix, min_samples)
    eps_candidates = _build_dbscan_eps_candidates(estimated_eps, k_distances)
    candidate_results = [
        _evaluate_dbscan_candidate(X_matrix, min_samples, candidate)
        for candidate in eps_candidates
    ]

    fallback_result = next(
        (candidate for candidate in candidate_results if np.isclose(candidate['eps'], estimated_eps)),
        candidate_results[0] if candidate_results else None,
    )
    selected_result, selection_reason, selected_from_fallback, valid_candidates = _select_best_dbscan_candidate(
        candidate_results,
        fallback_result,
    )

    if selected_result is None:
        return {
            'model': None,
            'labels': np.array([]),
            'eps': None,
            'estimated_eps': estimated_eps,
            'min_samples': min_samples,
            'cluster_labels': [],
            'cluster_count': 0,
            'noise_count': 0,
            'silhouette_score': None,
            'k_distances': k_distances,
            'eps_candidates': candidate_results,
            'selection_reason': selection_reason,
            'selected_from_fallback': selected_from_fallback,
        }

    return {
        'model': selected_result['model'],
        'labels': selected_result['labels'],
        'eps': selected_result['eps'],
        'estimated_eps': estimated_eps,
        'min_samples': min_samples,
        'cluster_labels': selected_result['cluster_labels'],
        'cluster_count': selected_result['cluster_count'],
        'noise_count': selected_result['noise_count'],
        'silhouette_score': selected_result['silhouette_score'],
        'k_distances': k_distances,
        'eps_candidates': candidate_results,
        'selected_candidate': selected_result,
        'valid_candidate_count': len(valid_candidates),
        'selection_reason': selection_reason,
        'selected_from_fallback': selected_from_fallback,
    }




def graficar_Kmeans(X, model, title="Clusters K-Means"):
    """Muestra un gráfico de clusters KMeans en espacio PCA de dos dimensiones.

    Args:
        X: dataframe o array con las características de entrada.
        model: modelo KMeans entrenado.
        title (str, optional): título del gráfico.

    Returns:
        None: muestra un scatter plot de clusters y centroides.
    """
    pca = PCA(n_components=2)

    X_pca = pca.fit_transform(X)
    centroids_pca = pca.transform(model.cluster_centers_)

    plt.figure(figsize=(8,6))

    plt.scatter(
        X_pca[:,0],
        X_pca[:,1],
        c=model.labels_,
        cmap='viridis',
        alpha=0.6
    )

    plt.scatter(
        centroids_pca[:,0],
        centroids_pca[:,1],
        marker='X',
        s=300,
        linewidths=2,
        edgecolors='black',
        label='Centroides'
    )

    plt.title(title)
    plt.legend()
    plt.show()


def graficar_dbscan(X, labels):
    """Visualiza los clusters encontrados por DBSCAN en dos dimensiones con PCA.

    Args:
        X (pd.DataFrame or np.ndarray): características de entrada.
        labels (np.ndarray): etiquetas de cluster producidas por DBSCAN.

    Returns:
        None: muestra un scatter plot de los clusters en PCA.
    """
    # Reducir dimensiones
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # Clusters encontrados
    unique_labels = np.unique(labels)

    plt.figure(figsize=(10, 6))

    for label in unique_labels:
        mask = labels == label

        # Ruido
        if label == -1:
            plt.scatter(
                X_pca[mask, 0],
                X_pca[mask, 1],
                marker='x',
                label='Ruido',
                alpha=0.6
            )
        else:
            plt.scatter(
                X_pca[mask, 0],
                X_pca[mask, 1],
                label=f'Cluster {label}',
                alpha=0.7
            )

    plt.title("Clusters generados por DBSCAN (PCA)")
    plt.xlabel("Componente Principal 1")
    plt.ylabel("Componente Principal 2")
    plt.legend()
    plt.grid(True)
    plt.show()    

def graficar_funciones_no_supervisadas(sil_kv, X, db_labv):
    """Grafica los resultados de KMeans para un conjunto de datos y modelo entrenado.

    Args:
        sil_kv: modelo KMeans entrenado o estrategia de selección previa.
        X: dataframe o array con las características de entrada.

    Returns:
        None: muestra el gráfico de clusters KMeans.
    """
    graficar_Kmeans(X,sil_kv)
    graficar_dbscan(X,db_labv)


def crear_tabla_no_supervisada(kmeans_reports, dbscan_reports):
    import pandas as pd

    filas = []
    for i in range(8):
        kr = kmeans_reports[i]
        dr = dbscan_reports[i]

        strategy_sil = kr['strategies']['silhouette']
        strategy_elb = kr['strategies']['elbow']

        k_inertia = strategy_sil['inertia'] if strategy_sil['inertia'] is not None else strategy_elb['inertia']
        k_sil = strategy_sil['silhouette_score'] if strategy_sil['silhouette_score'] is not None else strategy_elb['silhouette_score']

        filas.append([
            f'v{i+1}',
            round(k_inertia, 4) if k_inertia is not None else None,
            round(k_sil, 4) if k_sil is not None else None,
            dr['cluster_count'],
            dr['noise_count'],
            round(dr['silhouette_score'], 4) if dr['silhouette_score'] is not None else None,
        ])

    columnas = pd.MultiIndex.from_tuples([
        ('', 'Versión'),
        ('KMeans', 'Inercia'),
        ('KMeans', 'Silhouette'),
        ('DBSCAN', 'Clusters'),
        ('DBSCAN', 'Ruido'),
        ('DBSCAN', 'Silhouette'),
    ])

    return pd.DataFrame(filas, columns=columnas)

def graficar_silhouette_por_algoritmo(kmeans_reports, dbscan_reports):
    """ grafica el silhouette score de KMeans y DBSCAN para las 8 variaciones en barras

    Args:
        kmeans_reports (list[dict]): lista con los 8 reportes de train_kmeans_models
        dbscan_reports (list[dict]): lista con los 8 reportes de train_dbscan_model

    Returns:
        None: muestra la grafica con plt.show()
    """
    kmeans_sils = [r['strategies']['silhouette']['silhouette_score'] for r in kmeans_reports]
    dbscan_sils = [r['silhouette_score'] for r in dbscan_reports]

    x = np.arange(8)
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width/2, kmeans_sils, width, label='KMeans', color='#1565C0', edgecolor='black')
    ax.bar(x + width/2, dbscan_sils, width, label='DBSCAN', color='#E65100', edgecolor='black')

    ax.set_xticks(x)
    ax.set_xticklabels([f'v{i+1}' for i in range(8)])
    ax.set_xlabel('Variación')
    ax.set_ylabel('Silhouette Score')
    ax.set_title('Silhouette Score por variación - KMeans vs DBSCAN')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()


# Uso:

if __name__ == "__main__":
    pass

    #variacion 1
    printeo("Ejecutando variacion 1 supervisada...")
    arbolv1, knnv1, svmv1,modelov1, Columna_xv1, Xv1, yv1, knn_modelv1, red_neuronalv1 = variacion_supervisada(v1)    
    # #variacion 2
    # printeo("Ejecutando variacion 2 supervisada...")
    # arbolv2, knnv2, svmv2,modelov2, Columna_xv2, Xv2, yv2, knn_modelv2, red_neuronalv2 = variacion_supervisada(v2)
    # #variacion 3
    # printeo("Ejecutando variacion 3 supervisada...")
    # arbolv3, knnv3, svmv3,modelov3, Columna_xv3, Xv3, yv3, knn_modelv3, red_neuronalv3 = variacion_supervisada(v3)
    # #variacion 4
    # printeo("Ejecutando variacion 4 supervisada...")
    # arbolv4, knnv4, svmv4,modelov4, Columna_xv4, Xv4, yv4, knn_modelv4, red_neuronalv4 = variacion_supervisada(v4)
    # #variacion 5
    # printeo("Ejecutando variacion 5 supervisada...")
    # arbolv5, knnv5, svmv5,modelov5, Columna_xv5, Xv5, yv5, knn_modelv5, red_neuronalv5 = variacion_supervisada(v5)

    # #variacion 6
    # printeo("Ejecutando variacion 6 supervisada...")
    # arbolv6, knnv6, svmv6,modelov6, Columna_xv6, Xv6, yv6, knn_modelv6, red_neuronalv6 = variacion_supervisada(v6)
    # #variacion 7
    # printeo("Ejecutando variacion 7 supervisada...")
    # arbolv7, knnv7, svmv7,modelov7, Columna_xv7, Xv7, yv7, knn_modelv7, red_neuronalv7 = variacion_supervisada(v7)
    # #variacion 8
    # printeo("Ejecutando variacion 8 supervisada...")
    # arbolv8, knnv8, svmv8,modelov8, Columna_xv8, Xv8, yv8, knn_modelv8, red_neuronalv8 = variacion_supervisada(v8)
    
    #graficar comparacion de las 8 variaciones en tabla panda
    # tabla=grafico_comparacion_supervisada(arbolv1, knnv1, svmv1, red_neuronalv1, arbolv2, knnv2, svmv2, red_neuronalv2, arbolv3, knnv3, svmv3, red_neuronalv3, arbolv4, knnv4, svmv4, red_neuronalv4, 
    #                     arbolv5, knnv5, svmv5, red_neuronalv5, arbolv6, knnv6, svmv6, red_neuronalv6, arbolv7, knnv7, svmv7, red_neuronalv7, arbolv8, knnv8, svmv8, red_neuronalv8)
    # print(tabla)
    # graficar_tabla_comparacion(tabla)
    
    
    # # variables para graficar comparacion de metricas entre las 8 variaciones para cada algoritmo
    # arboles = [arbolv1, arbolv2, arbolv3, arbolv4, arbolv5, arbolv6, arbolv7, arbolv8]
    # knns    = [knnv1, knnv2, knnv3, knnv4, knnv5, knnv6, knnv7, knnv8]
    # svms    = [svmv1, svmv2, svmv3, svmv4, svmv5, svmv6, svmv7, svmv8]
    # redes   = [red_neuronalv1, red_neuronalv2, red_neuronalv3, red_neuronalv4,
    #         red_neuronalv5, red_neuronalv6, red_neuronalv7, red_neuronalv8]
    
    
    # graficar comparacion de metricas entre las 8 variaciones para cada algoritmo
    #printeo("Graficando comparacion de F1-Score máximo por algoritmo...")
    # graficar_f1_maximo_supervisado(arboles, knns, svms, redes)
    
    #printeo("Graficando comparacion de F1-Score medio por normalizacion y algoritmo...")
    #graficar_f1_medio_por_normalizacion(arboles, knns, svms, redes)
    
    # printeo("Graficando comparacion de F1-Score medio por outliers y algoritmo...")
    # graficar_f1_medio_por_outliers(arboles, knns, svms, redes)
    
    #grafico
    # printeo("Graficando comparacion de F1-Score medio por balanceo y algoritmo...")
    # graficar_f1_medio_por_balanceo(arboles, knns, svms, redes)
    
    # printeo("Graficando curva ROC y Precision-Recall para la variacion 8...")
    # graficar_roc_pr_v8(Xv8, yv8)
    
    #graficar arbol, knn y svm de la variacion 1 a la 8
    
    # printeo("Graficos de arbol, knn y svm para variacion 1")
    # graficar_funciones_supervisadas(knn_modelv1, modelov1, Columna_xv1, Xv1, yv1)
    # printeo("Graficos de arbol, knn y svm para variacion 2")
    # graficar_funciones_supervisadas(knn_modelv2, modelov2, Columna_xv2, Xv2, yv2)
    # printeo("Graficos de arbol, knn y svm para variacion 3")
    # graficar_funciones_supervisadas(knn_modelv3, modelov3, Columna_xv3, Xv3, yv3)
    # printeo("Graficos de arbol, knn y svm para variacion 4")
    # graficar_funciones_supervisadas(knn_modelv4, modelov4, Columna_xv4, Xv4, yv4)
    # printeo("Graficos de arbol, knn y svm para variacion 5")
    # graficar_funciones_supervisadas(knn_modelv5, modelov5, Columna_xv5, Xv5, yv5)
    # printeo("Graficos de arbol, knn y svm para variacion 6")
    # graficar_funciones_supervisadas(knn_modelv6, modelov6, Columna_xv6, Xv6, yv6)
    # printeo("Graficos de arbol, knn y svm para variacion 7")
    # graficar_funciones_supervisadas(knn_modelv7, modelov7, Columna_xv7, Xv7, yv7)
    # printeo("Graficos de arbol, knn y svm para variacion 8")
    # graficar_funciones_supervisadas(knn_modelv8, modelov8, Columna_xv8, Xv8, yv8)
    
    
    
    
    
    # flujo KMeans
        # flujo no supervisado
        
    # printeo("Ejecutando variacion 1 no supervisada...")
    # (kmeansv1, km_elbowv1, km_silv1, elbow_kv1, sil_kv1,
    # dbscanv1, db_modv1, epsv1, db_labv1) = variacion_no_supervisada(v1)

    # printeo("Ejecutando variacion 2 no supervisada...")
    # (kmeansv2, km_elbowv2, km_silv2, elbow_kv2, sil_kv2,
    #  dbscanv2, db_modv2, epsv2, db_labv2) = variacion_no_supervisada(v2)

    # printeo("Ejecutando variacion 3 no supervisada...")
    # (kmeansv3, km_elbowv3, km_silv3, elbow_kv3, sil_kv3,
    #  dbscanv3, db_modv3, epsv3, db_labv3) = variacion_no_supervisada(v3)

    # printeo("Ejecutando variacion 4 no supervisada...")
    # (kmeansv4, km_elbowv4, km_silv4, elbow_kv4, sil_kv4,
    #  dbscanv4, db_modv4, epsv4, db_labv4) = variacion_no_supervisada(v4)

    # printeo("Ejecutando variacion 5 no supervisada...")
    # (kmeansv5, km_elbowv5, km_silv5, elbow_kv5, sil_kv5,
    #  dbscanv5, db_modv5, epsv5, db_labv5) = variacion_no_supervisada(v5)

    # printeo("Ejecutando variacion 6 no supervisada...")
    # (kmeansv6, km_elbowv6, km_silv6, elbow_kv6, sil_kv6,
    #  dbscanv6, db_modv6, epsv6, db_labv6) = variacion_no_supervisada(v6)

    # printeo("Ejecutando variacion 7 no supervisada...")
    # (kmeansv7, km_elbowv7, km_silv7, elbow_kv7, sil_kv7,
    #  dbscanv7, db_modv7, epsv7, db_labv7) = variacion_no_supervisada(v7)

    # printeo("Ejecutando variacion 8 no supervisada...")
    # (kmeansv8, km_elbowv8, km_silv8, elbow_kv8, sil_kv8,
    #  dbscanv8, db_modv8, epsv8, db_labv8) = variacion_no_supervisada(v8)


    # # comparacion de resultados entre las 8 variaciones para cada algoritmo no supervisado
    # kmeans_reports = [kmeansv1, kmeansv2, kmeansv3, kmeansv4,
    #                 kmeansv5, kmeansv6, kmeansv7, kmeansv8]
    # dbscan_reports = [dbscanv1, dbscanv2, dbscanv3, dbscanv4,
    #                 dbscanv5, dbscanv6, dbscanv7, dbscanv8]

    # tabla_ns = crear_tabla_no_supervisada(kmeans_reports, dbscan_reports)
    # print(tabla_ns)
    # graficar_tabla_comparacion(tabla_ns)


    # kmeans_reports = [kmeansv1, kmeansv2, kmeansv3, kmeansv4,
    #                 kmeansv5, kmeansv6, kmeansv7, kmeansv8]
    # dbscan_reports = [dbscanv1, dbscanv2, dbscanv3, dbscanv4,
    #                 dbscanv5, dbscanv6, dbscanv7, dbscanv8]
    # graficar_silhouette_por_algoritmo(kmeans_reports, dbscan_reports)

    #graficar Kmeans y DBSACN de la variacion 1 a la 8
    # printeo("Graficos de Kmeans y DBSACN para variacion 1")
    # graficar_funciones_no_supervisadas(km_silv1, Xv1, db_labv1)
    # printeo("Graficos de Kmeans y DBSACN para variacion 2")
    # graficar_funciones_no_supervisadas(km_silv2, Xv2, db_labv2)
    # printeo("Graficos de Kmeans y DBSACN para variacion 3")
    # graficar_funciones_no_supervisadas(km_silv3, Xv3, db_labv3)
    # printeo("Graficos de Kmeans y DBSACN para variacion 4")
    # graficar_funciones_no_supervisadas(km_silv4, Xv4, db_labv4)
    # printeo("Graficos de Kmeans y DBSACN para variacion 5")
    # graficar_funciones_no_supervisadas(km_silv5, Xv5, db_labv5)
    # printeo("Graficos de Kmeans y DBSACN para variacion 6")
    # graficar_funciones_no_supervisadas(km_silv6, Xv6, db_labv6)
    # printeo("Graficos de Kmeans y DBSACN para variacion 7")
    # graficar_funciones_no_supervisadas(km_silv7, Xv7, db_labv7)
    # printeo("Graficos de Kmeans y DBSACN para variacion 8")
    # graficar_funciones_no_supervisadas(km_silv8, Xv8, db_labv8)