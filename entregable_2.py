import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Cargar dataset
df = pd.read_csv('wine_quality_merged.csv')
df = df.sample(n=5150, random_state=41).reset_index(drop=True)

# CC: conversion categorica a numerica
df['type'] = df['type'].map({'red': 1, 'white': 0})
print(df.head())

def prepare_data(df, escalado, outliers, balanceo):
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
    filas = []
    for i, params in enumerate(variaciones, 1):
        df_proc, X, y = prepare_data(df, **params)
        clases, counts = np.unique(y, return_counts=True)
        dist = dict(zip(clases, counts))
        filas.append({
            'Version': i,
            'CC': 'si',
            'ED': params['escalado'],
            'Outliers': params['outliers'],
            'Balanceo': params['balanceo'],
            'Muestras': X.shape[0],
            'Features': X.shape[1],
            'Distribucion': dist
        })
    return pd.DataFrame(filas)

print("\n\n",generar_tabla_variaciones(df, variaciones))

# Guardar cada variacion en una variable
v1 = prepare_data(df, **variaciones[0])  #caso 1(df_proc, X, y)
v2 = prepare_data(df, **variaciones[1]) #caso 2
v3 = prepare_data(df, **variaciones[2]) #caso 3
v4 = prepare_data(df, **variaciones[3]) #caso 4
v5 = prepare_data(df, **variaciones[4]) #caso 5
v6 = prepare_data(df, **variaciones[5]) #caso 6
v7 = prepare_data(df, **variaciones[6]) #caso 7
v8 = prepare_data(df, **variaciones[7]) #caso 8

print("\n\nVERIFICAR UNA VARIACIÓN PROCESADA (ejemplo v1):")
v1[0].head() # Mostrar el DataFrame procesado de la última variación

def metricas(y_test, y_pred):
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
    #Separacion de datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=41)
    
    #Creacion del modelo
    model = SVC(kernel='rbf', gamma="auto", C=1.0)  # C=1.0 (Equilibrado)
    model.fit(X_train, y_train)

    #Predicciones
    y_pred = model.predict(X_test)

    #Evaluacion del modelo
    metrica= metricas(y_test, y_pred)
    
    return metrica



def redes_neuronales():
    pass




def grafico_comparacion(arbolv1, knnv1, svmv1, arbolv2, knnv2, svmv2, arbolv3, knnv3, svmv3, arbolv4, knnv4, svmv4, 
                        arbolv5, knnv5, svmv5, arbolv6, knnv6, svmv6, arbolv7, knnv7, svmv7, arbolv8, knnv8, svmv8):    
    # Definir los algoritmos y métricas
    algoritmos = ['ÁRBOLES DE DECISIÓN', 'K VECINOS MAS CERCANOS (KNN)', 'MAQUINAS DE VECTORES DE SOPORTE (SVM)' ]
    metricas = ['ACCURACY', 'PRECISION', 'RECALL', 'F1-SCORE']

    # Crear MultiIndex para las columnas
    columnas_fijas = [('#', ''), ('Normalización', ''), ('OUTLIERS', ''), ('BALANCEO', '')]
    columnas_metricas = [(alg, met) for alg in algoritmos for met in metricas]

    multi_index = pd.MultiIndex.from_tuples(columnas_fijas + columnas_metricas)

    # Datos de las filas
    datos = [
        [1, 'CC(SI) y ED(NO)', 'NO', 'NO', arbolv1[0], arbolv1[1], arbolv1[2], arbolv1[3],
         knnv1[0], knnv1[1], knnv1[2], knnv1[3], svmv1[0], svmv1[1], svmv1[2], svmv1[3]],
        
        [2, 'CC(SI) y ED(NO)', 'NO', 'SI', arbolv2[0], arbolv2[1], arbolv2[2], arbolv2[3], 
         knnv2[0], knnv2[1], knnv2[2], knnv2[3], svmv2[0], svmv2[1], svmv2[2], svmv2[3]],
        
        [3, 'CC(SI) y ED(NO)', 'SI', 'NO', arbolv3[0], arbolv3[1], arbolv3[2], arbolv3[3], 
         knnv3[0], knnv3[1], knnv3[2], knnv3[3],svmv3[0], svmv3[1], svmv3[2], svmv3[3]],
        
        [4, 'CC(SI) y ED(NO)', 'SI', 'SI', arbolv4[0], arbolv4[1], arbolv4[2], arbolv4[3], 
         knnv4[0], knnv4[1], knnv4[2], knnv4[3], svmv4[0], svmv4[1], svmv4[2], svmv4[3]],
        
        [5, 'CC(SI) y ED(SI)', 'NO', 'NO', arbolv5[0], arbolv5[1], arbolv5[2], arbolv5[3], 
         knnv5[0], knnv5[1], knnv5[2], knnv5[3], svmv5[0], svmv5[1], svmv5[2], svmv5[3]],
        
        [6, 'CC(SI) y ED(SI)', 'NO', 'SI', arbolv6[0], arbolv6[1], arbolv6[2], arbolv6[3], 
         knnv6[0], knnv6[1], knnv6[2], knnv6[3], svmv6[0], svmv6[1], svmv6[2], svmv6[3]],
        
        [7, 'CC(SI) y ED(SI)', 'SI', 'NO', arbolv7[0], arbolv7[1], arbolv7[2], arbolv7[3], 
         knnv7[0], knnv7[1], knnv7[2], knnv7[3], svmv7[0], svmv7[1], svmv7[2], svmv7[3]],
        
        [8, 'CC(SI) y ED(SI)', 'SI', 'SI', arbolv8[0], arbolv8[1], arbolv8[2], arbolv8[3], 
         knnv8[0], knnv8[1], knnv8[2], knnv8[3], svmv8[0], svmv8[1], svmv8[2], svmv8[3]],
    ]

    df = pd.DataFrame(datos, columns=multi_index)
    print(df)
    
    return df


def variacion_no_supervisada(variacion):
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
    print(f"Metricas svm: {metrica_svm}\n")
    svm_metrics = [metrica_svm['accuracy'],metrica_svm["precision_weighted"], 
             metrica_svm['recall_weighted'], metrica_svm['f1_weighted']]
    
    return arbol, knn, svm_metrics, modelo, Columna_x, X, y, knn_model



from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def graficar_arbol(modelo, feature_names):
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



def graficar_funciones(knn,modelo, Columna_x, X, y):
    graficar_arbol(modelo, Columna_x)
    graficar_knn(X, y, knn)







# Uso:










if __name__ == "__main__":
    
    
    #variacion 1
    arbolv1, knnv1, svmv1,modelov1, Columna_xv1, Xv1, yv1, knn_modelv1 = variacion_no_supervisada(v1)
    
    #variacion 2
    arbolv2, knnv2, svmv2,modelov2, Columna_xv2, Xv2, yv2, knn_modelv2 = variacion_no_supervisada(v2)
    
    #variacion 3
    arbolv3, knnv3, svmv3,modelov3, Columna_xv3, Xv3, yv3, knn_modelv3 = variacion_no_supervisada(v3)
    
    #variacion 4
    arbolv4, knnv4, svmv4,modelov4, Columna_xv4, Xv4, yv4, knn_modelv4 = variacion_no_supervisada(v4)
    
    #variacion 5
    arbolv5, knnv5, svmv5,modelov5, Columna_xv5, Xv5, yv5, knn_modelv5 = variacion_no_supervisada(v5)
    
    #variacion 6
    arbolv6, knnv6, svmv6,modelov6, Columna_xv6, Xv6, yv6, knn_modelv6 = variacion_no_supervisada(v6)
    
    #variacion 7
    arbolv7, knnv7, svmv7,modelov7, Columna_xv7, Xv7, yv7, knn_modelv7 = variacion_no_supervisada(v7)
    
    #variacion 8
    arbolv8, knnv8, svmv8,modelov8, Columna_xv8, Xv8, yv8, knn_modelv8 = variacion_no_supervisada(v8)
    
    grafico_comparacion(arbolv1, knnv1, svmv1, arbolv2, knnv2, svmv2, arbolv3, knnv3, svmv3, arbolv4, knnv4, svmv4, 
                        arbolv5, knnv5, svmv5, arbolv6, knnv6, svmv6, arbolv7, knnv7, svmv7, arbolv8, knnv8, svmv8)
    
    #graficar arbol y knn de la variacion 1
    graficar_funciones(knn_modelv1, modelov1, Columna_xv1, Xv1, yv1)

    #No supervisado

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score  
from kneed import KneeLocator

def K_means(X,y):
    inertia = []
    K_range = range(1, 11)

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=41, n_init=10)
        kmeans.fit(X)
        inertia.append(kmeans.inertia_)   # Métrica Inercia.

    
    kl = KneeLocator(K_range, inertia, curve='convex', direction='decreasing')

    print(f"El valor de K óptimo calculado automáticamente es: {kl.elbow}")

    
    kl.plot_knee()
    plt.show()

    silhouette_scores = []

    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k, random_state=41, n_init=10)
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)
        silhouette_scores.append(score)
    pass
    
def DBSCAN(X,y):
    pass