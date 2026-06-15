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
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans, DBSCAN as SklearnDBSCAN
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







#No supervisado

from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score  
try:
    from kneed import KneeLocator
except ImportError:
    KneeLocator = None


def prepare_unsupervised_data(df, escalado, outliers, balanceo):
    """Prepare the feature matrix for unsupervised models without using quality as input."""
    df_processed = df.copy()
    X = df_processed.drop('quality', axis=1)
    y = df_processed['quality']
    source_indices = np.asarray(X.index)

    if outliers == 'no':
        iso_forest = IsolationForest(contamination=0.05, random_state=41)
        outlier_mask = iso_forest.fit_predict(X) == 1
        X = X[outlier_mask].copy()
        y = y[outlier_mask].copy()
        source_indices = source_indices[outlier_mask]

    scaler = None
    if escalado == 'si':
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

    if balanceo == 'si':
        smote = SMOTE(random_state=41, k_neighbors=4)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        X = pd.DataFrame(X_resampled, columns=X.columns)
        synthetic_count = len(X) - len(source_indices)
        if synthetic_count > 0:
            source_indices = np.concatenate([
                source_indices,
                np.full(synthetic_count, -1, dtype=int),
            ])
        y = y_resampled

    return {
        'X': X,
        'y': y,
        'scaler': scaler,
        'feature_columns': list(X.columns),
        'source_indices': source_indices,
    }

def _compute_kmeans_inertia(X, k_values, random_state=41):
    inertia = []
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        kmeans.fit(X)
        inertia.append(kmeans.inertia_)
    return inertia


def _select_k_by_elbow(k_values, inertia, fallback_k):
    if KneeLocator is None or len(k_values) < 2:
        return fallback_k

    locator = KneeLocator(k_values, inertia, curve='convex', direction='decreasing')
    return locator.elbow if locator.elbow is not None else fallback_k


def _compute_silhouette_scores(X, k_values, random_state=41):
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


def train_kmeans_models(X, random_state=41, k_range=range(1, 11), silhouette_k_range=range(2, 11)):
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
        'k_values': k_values,
        'silhouette_k_values': silhouette_values,
        'inertia': inertia,
        'silhouette_scores': silhouette_scores,
        'selected_k_elbow': elbow_k,
        'selected_k_silhouette': silhouette_k,
        'selected_silhouette_score': silhouette_best_score,
        'strategies': {
            'elbow': {
                'selected_k': elbow_k,
                'inertia': elbow_model.inertia_ if elbow_model is not None else None,
                'silhouette_score': elbow_silhouette,
                'labels': elbow_labels,
                'model': elbow_model,
            },
            'silhouette': {
                'selected_k': silhouette_k,
                'inertia': silhouette_model.inertia_ if silhouette_model is not None else None,
                'silhouette_score': silhouette_model_score,
                'labels': silhouette_labels,
                'model': silhouette_model,
            },
        },
    }


def K_means(X, y):
    return train_kmeans_models(X)


def DBSCAN(X, y):
    return train_dbscan_model(X)


def _as_numpy_matrix(X):
    if isinstance(X, pd.DataFrame):
        return X.to_numpy()
    return np.asarray(X)


def _safe_dbscan_silhouette_score(X, labels, noise_label=-1):
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


def train_dbscan_model(X, min_samples=None):
    X_matrix = _as_numpy_matrix(X)
    if X_matrix.size == 0:
        return {
            'model': None,
            'labels': np.array([]),
            'eps': None,
            'min_samples': None,
            'cluster_labels': [],
            'cluster_count': 0,
            'noise_count': 0,
            'silhouette_score': None,
            'k_distances': np.array([]),
        }

    n_samples, feature_count = X_matrix.shape
    if min_samples is None:
        min_samples = max(3, feature_count + 1)
    min_samples = int(min_samples)
    min_samples = min(min_samples, n_samples)
    min_samples = max(2, min_samples) if n_samples >= 2 else n_samples

    eps, k_distances = _estimate_dbscan_eps(X_matrix, min_samples)
    model = SklearnDBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X_matrix)

    cluster_labels = sorted(int(label) for label in np.unique(labels) if label != -1)
    cluster_count = len(cluster_labels)
    noise_count = int(np.sum(labels == -1))
    dbscan_silhouette = _safe_dbscan_silhouette_score(X_matrix, labels)

    return {
        'model': model,
        'labels': labels,
        'eps': eps,
        'min_samples': min_samples,
        'cluster_labels': cluster_labels,
        'cluster_count': cluster_count,
        'noise_count': noise_count,
        'silhouette_score': dbscan_silhouette,
        'k_distances': k_distances,
    }


def _kmeans_preprocessing_label(params):
    return f"CC(si), ED({params['escalado']}), outliers({params['outliers']}), balanceo({params['balanceo']})"


def _fit_kmeans_variant(df, params, random_state=41):
    prepared = prepare_unsupervised_data(df, **params)
    report = train_kmeans_models(prepared['X'], random_state=random_state)

    best_strategy = report['strategies']['silhouette']
    if best_strategy['model'] is None:
        best_strategy = report['strategies']['elbow']

    return {
        'params': params,
        'label': _kmeans_preprocessing_label(params),
        'prepared': prepared,
        'report': report,
        'prediction_strategy': best_strategy,
    }


def _build_kmeans_summary_row(index, variant_result):
    report = variant_result['report']
    strategy = report['strategies']['silhouette']
    if strategy['model'] is None:
        strategy = report['strategies']['elbow']

    return {
        'Version': index,
        'Preprocessing': variant_result['label'],
        'KMeans inertia': round(strategy['inertia'], 4) if strategy['inertia'] is not None else None,
        'KMeans silhouette_score': round(strategy['silhouette_score'], 4) if strategy['silhouette_score'] is not None else None,
    }


def _build_dbscan_summary_row(index, variant_result):
    report = variant_result['dbscan_report']
    labels = np.asarray(report['labels'])
    unique_labels = sorted(int(label) for label in np.unique(labels))

    return {
        'Version': index,
        'Preprocessing': variant_result['label'],
        'DBSCAN eps': round(report['eps'], 4) if report['eps'] is not None else None,
        'DBSCAN min_samples': int(report['min_samples']) if report['min_samples'] is not None else None,
        'DBSCAN labels': unique_labels,
        'DBSCAN cluster_count': int(report['cluster_count']),
        'DBSCAN noise_count': int(report['noise_count']),
        'DBSCAN silhouette_score': round(report['silhouette_score'], 4) if report['silhouette_score'] is not None else None,
    }


def _build_kmeans_prediction_cases(df, variant_result, strategy_name, case_indices=(0, 1, 2)):
    prepared = variant_result['prepared']
    strategy = variant_result['report']['strategies'][strategy_name]
    model = strategy['model']
    if model is None:
        return []

    cases = []
    for case_number, row_index in enumerate(case_indices, 1):
        raw_row = df.iloc[[row_index]][prepared['feature_columns']].copy()
        if prepared['scaler'] is not None:
            row_for_model = pd.DataFrame(
                prepared['scaler'].transform(raw_row),
                columns=prepared['feature_columns'],
                index=raw_row.index,
            )
        else:
            row_for_model = raw_row.copy()

        predicted_cluster = int(model.predict(row_for_model)[0])
        centroid_distances = model.transform(row_for_model)[0].round(4).tolist()

        cases.append({
            'strategy': strategy_name,
            'case': case_number,
            'row_index': int(df.index[row_index]),
            'predicted_cluster': predicted_cluster,
            'distances_to_centroids': centroid_distances,
        })

    return cases


def _build_dbscan_case_reports(df, variant_result, case_count=3):
    prepared = variant_result['prepared']
    labels = np.asarray(variant_result['dbscan_report']['labels'])
    source_indices = np.asarray(prepared['source_indices'])

    valid_positions = np.flatnonzero(source_indices >= 0)
    selected_positions = list(valid_positions[:case_count])

    if len(selected_positions) < case_count:
        for position in range(len(labels)):
            if position not in selected_positions and source_indices[position] >= 0:
                selected_positions.append(position)
            if len(selected_positions) == case_count:
                break

    cases = []
    for case_number, position in enumerate(selected_positions[:case_count], 1):
        cases.append({
            'case': case_number,
            'source_row_index': int(source_indices[position]),
            'prepared_position': int(position),
            'dbscan_label': int(labels[position]),
            'is_noise': bool(labels[position] == -1),
        })

    return cases


def generar_reporte_kmeans(df, variaciones, case_indices=(0, 1, 2)):
    variant_results = []
    summary_rows = []

    for index, params in enumerate(variaciones, 1):
        variant_result = _fit_kmeans_variant(df, params)
        elbow_predictions = _build_kmeans_prediction_cases(df, variant_result, 'elbow', case_indices=case_indices)
        silhouette_predictions = _build_kmeans_prediction_cases(df, variant_result, 'silhouette', case_indices=case_indices)
        variant_result['predictions_by_strategy'] = {
            'elbow': elbow_predictions,
            'silhouette': silhouette_predictions,
        }
        variant_result['predictions'] = elbow_predictions + silhouette_predictions
        variant_results.append(variant_result)
        summary_rows.append(_build_kmeans_summary_row(index, variant_result))

    summary_df = pd.DataFrame(summary_rows)
    return summary_df, variant_results


def generar_reporte_no_supervisado(df, variaciones, case_indices=(0, 1, 2)):
    variant_results = []
    summary_rows = []

    for index, params in enumerate(variaciones, 1):
        variant_result = _fit_kmeans_variant(df, params)
        dbscan_report = train_dbscan_model(variant_result['prepared']['X'])
        variant_result['dbscan_report'] = dbscan_report
        variant_result['dbscan_predictions'] = _build_dbscan_case_reports(df, variant_result, case_count=len(case_indices))
        variant_results.append(variant_result)
        summary_rows.append({
            **_build_kmeans_summary_row(index, variant_result),
            **_build_dbscan_summary_row(index, variant_result),
        })

    summary_df = pd.DataFrame(summary_rows)
    return summary_df, variant_results


def graficar_silhouette_kmeans(variant_results):
    labels = [f'V{i}' for i in range(1, len(variant_results) + 1)]
    scores = []

    for variant_result in variant_results:
        strategy = variant_result['report']['strategies']['silhouette']
        if strategy['silhouette_score'] is None:
            strategy = variant_result['report']['strategies']['elbow']
        scores.append(strategy['silhouette_score'] if strategy['silhouette_score'] is not None else 0)

    plt.figure(figsize=(10, 5))
    plt.bar(labels, scores, color='steelblue')
    plt.title('KMeans silhouette score por variación')
    plt.xlabel('Variación')
    plt.ylabel('Silhouette score')
    plt.ylim(0, 1)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()


def graficar_silhouette_dbscan(variant_results):
    labels = [f'V{i}' for i in range(1, len(variant_results) + 1)]
    scores = []

    for variant_result in variant_results:
        score = variant_result['dbscan_report']['silhouette_score']
        scores.append(score if score is not None else 0)

    plt.figure(figsize=(10, 5))
    plt.bar(labels, scores, color='mediumpurple')
    plt.title('DBSCAN silhouette score por variación')
    plt.xlabel('Variación')
    plt.ylabel('Silhouette score')
    plt.ylim(-1, 1)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()


def graficar_elbow_comparativo_kmeans(variant_results):
    if not variant_results:
        return

    fig, ax = plt.subplots(figsize=(12, 7))
    cmap = plt.get_cmap('tab10')

    all_k_values = set()
    for index, variant_result in enumerate(variant_results, 1):
        report = variant_result['report']
        k_values = report['k_values']
        inertia = report['inertia']
        selected_k = report['selected_k_elbow']
        color = cmap((index - 1) % 10)

        all_k_values.update(k_values)
        legend_label = f"V{index}" if selected_k is None else f"V{index} (K={selected_k})"

        ax.plot(
            k_values,
            inertia,
            marker='o',
            linewidth=1.8,
            markersize=4,
            label=legend_label,
            color=color,
            alpha=0.85,
        )

        if selected_k in k_values:
            selected_index = k_values.index(selected_k)
            ax.scatter(
                [selected_k],
                [inertia[selected_index]],
                color=color,
                s=55,
                zorder=5,
                edgecolors='black',
                linewidths=0.5,
            )

    ax.set_title('Comparativa de curvas elbow de KMeans por variación')
    ax.set_xlabel('K values')
    ax.set_ylabel('Inertia')
    ax.set_xticks(sorted(all_k_values))
    ax.grid(axis='y', alpha=0.3)
    ax.legend(ncol=2, fontsize=8)

    plt.tight_layout()
    plt.show()


def graficar_clusters_pca_kmeans(variant_results):
    if not variant_results:
        return

    n_rows = len(variant_results)
    fig, axes = plt.subplots(n_rows, 2, figsize=(18, 4 * n_rows), sharex=True, sharey=True)
    if n_rows == 1:
        axes = np.array([axes])

    strategy_info = [
        ('elbow', 'Elbow'),
        ('silhouette', 'Silhouette'),
    ]
    cmap = plt.get_cmap('tab10')

    for row_index, variant_result in enumerate(variant_results):
        prepared = variant_result['prepared']
        X_variant = prepared['X']
        X_variant_matrix = X_variant.to_numpy() if isinstance(X_variant, pd.DataFrame) else np.asarray(X_variant)
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_variant_matrix)

        for col_index, (strategy_name, strategy_title) in enumerate(strategy_info):
            ax = axes[row_index, col_index]
            strategy = variant_result['report']['strategies'][strategy_name]
            model = strategy['model']
            labels = strategy['labels']

            if model is None or labels is None:
                ax.set_axis_off()
                continue

            labels = np.asarray(labels)
            unique_labels = np.unique(labels)
            for cluster_id in unique_labels:
                cluster_mask = labels == cluster_id
                ax.scatter(
                    X_pca[cluster_mask, 0],
                    X_pca[cluster_mask, 1],
                    s=8,
                    alpha=0.65,
                    color=cmap(int(cluster_id) % 10),
                    edgecolors='none',
                )

            centroid_pca = pca.transform(model.cluster_centers_)
            ax.scatter(
                centroid_pca[:, 0],
                centroid_pca[:, 1],
                s=140,
                c='black',
                marker='X',
                edgecolors='white',
                linewidths=0.8,
                zorder=5,
            )

            selected_k = strategy['selected_k']
            ax.set_title(f"V{row_index + 1} - {strategy_title} (K={selected_k})", fontsize=9)
            if row_index == n_rows - 1:
                ax.set_xlabel('PC1')
            if col_index == 0:
                ax.set_ylabel('PC2')
            ax.grid(alpha=0.2)
            ax.tick_params(labelsize=7)

    fig.suptitle('KMeans clusters projected with PCA (8 variants x 2 strategies)', fontsize=14)
    plt.tight_layout(rect=[0, 0.02, 1, 0.98])
    plt.show()


def graficar_clusters_pca_dbscan(variant_results):
    if not variant_results:
        return

    n_rows = len(variant_results)
    fig, axes = plt.subplots(n_rows, 1, figsize=(14, 4 * n_rows), sharex=True, sharey=True)
    if n_rows == 1:
        axes = np.array([axes])

    cmap = plt.get_cmap('tab10')

    for row_index, variant_result in enumerate(variant_results):
        prepared = variant_result['prepared']
        X_variant = prepared['X']
        X_variant_matrix = X_variant.to_numpy() if isinstance(X_variant, pd.DataFrame) else np.asarray(X_variant)
        if X_variant_matrix.shape[1] >= 2:
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_variant_matrix)
        else:
            X_pca = np.column_stack([X_variant_matrix[:, 0], np.zeros(X_variant_matrix.shape[0])])

        report = variant_result['dbscan_report']
        labels = np.asarray(report['labels'])
        ax = axes[row_index]

        if labels.size == 0:
            ax.set_axis_off()
            continue

        unique_labels = sorted(np.unique(labels), key=lambda value: (value == -1, value))
        for cluster_id in unique_labels:
            cluster_mask = labels == cluster_id
            if cluster_id == -1:
                ax.scatter(
                    X_pca[cluster_mask, 0],
                    X_pca[cluster_mask, 1],
                    s=22,
                    alpha=0.9,
                    color='black',
                    marker='x',
                    label='Ruido',
                )
            else:
                ax.scatter(
                    X_pca[cluster_mask, 0],
                    X_pca[cluster_mask, 1],
                    s=18,
                    alpha=0.7,
                    color=cmap(int(cluster_id) % 10),
                    edgecolors='none',
                    label=f'Clúster {int(cluster_id)}',
                )

        ax.set_title(
            f"V{row_index + 1} - DBSCAN (eps={report['eps']:.3f}, min_samples={report['min_samples']})",
            fontsize=9,
        )
        if row_index == n_rows - 1:
            ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.grid(alpha=0.2)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, ncol=2, loc='best')

    fig.suptitle('Clústeres de DBSCAN proyectados con PCA (8 variantes)', fontsize=14)
    plt.tight_layout(rect=[0, 0.02, 1, 0.98])
    plt.show()


def mostrar_predicciones_kmeans(variant_results):
    for index, variant_result in enumerate(variant_results, 1):
        print(f"\nPredicciones KMeans - Variación {index}: {variant_result['label']}")
        grouped_predictions = variant_result.get('predictions_by_strategy', {})
        if not grouped_predictions:
            grouped_predictions = {}
            for case in variant_result.get('predictions', []):
                grouped_predictions.setdefault(case['strategy'], []).append(case)

        for strategy_name, cases in grouped_predictions.items():
            print(f"  Estrategia {strategy_name}:")
            for case in cases:
                print(
                    f"    Caso {case['case']} (fila {case['row_index']}): "
                    f"cluster={case['predicted_cluster']}, distancias={case['distances_to_centroids']}"
                )


def mostrar_predicciones_dbscan(variant_results):
    for index, variant_result in enumerate(variant_results, 1):
        print(f"\nPredicciones DBSCAN - Variación {index}: {variant_result['label']}")
        for case in variant_result.get('dbscan_predictions', []):
            print(
                f"  Caso {case['case']} (fila original {case['source_row_index']}): "
                f"label={case['dbscan_label']}, ruido={case['is_noise']}"
            )


def mostrar_analisis_resultados_no_supervisados(summary_df, variant_results):
    kmeans_scores = []
    dbscan_scores = []

    for variant_result in variant_results:
        kmeans_strategy = variant_result['report']['strategies']['silhouette']
        if kmeans_strategy['silhouette_score'] is None:
            kmeans_strategy = variant_result['report']['strategies']['elbow']
        if kmeans_strategy['silhouette_score'] is not None:
            kmeans_scores.append((variant_result['label'], kmeans_strategy['silhouette_score']))

        dbscan_score = variant_result['dbscan_report']['silhouette_score']
        if dbscan_score is not None:
            dbscan_scores.append((variant_result['label'], dbscan_score))

    best_kmeans = max(kmeans_scores, key=lambda item: item[1]) if kmeans_scores else None
    best_dbscan = max(dbscan_scores, key=lambda item: item[1]) if dbscan_scores else None
    invalid_dbscan = len(variant_results) - len(dbscan_scores)

    print("\nANÁLISIS RÁPIDO")
    if best_kmeans is not None:
        print(f"- Mejor KMeans silhouette: {best_kmeans[0]} ({best_kmeans[1]:.4f})")
    else:
        print("- Mejor KMeans silhouette: no disponible")

    if best_dbscan is not None:
        print(f"- Mejor DBSCAN silhouette: {best_dbscan[0]} ({best_dbscan[1]:.4f})")
    else:
        print("- Mejor DBSCAN silhouette: no disponible")

    print(f"- DBSCAN sin silhouette válida: {invalid_dbscan}/{len(variant_results)} variantes")


def ejecutar_flujo_kmeans(df, variaciones):
    print("\n\nFIGURA 3 - Resumen No Supervisado por variación")
    summary_df, variant_results = generar_reporte_no_supervisado(df, variaciones)
    print(summary_df)

    graficar_clusters_pca_kmeans(variant_results)
    graficar_clusters_pca_dbscan(variant_results)
    graficar_silhouette_kmeans(variant_results)
    graficar_silhouette_dbscan(variant_results)
    graficar_elbow_comparativo_kmeans(variant_results)
    mostrar_predicciones_kmeans(variant_results)
    mostrar_predicciones_dbscan(variant_results)
    mostrar_analisis_resultados_no_supervisados(summary_df, variant_results)

    return summary_df, variant_results


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
    
    # flujo KMeans
    ejecutar_flujo_kmeans(df, variaciones)
