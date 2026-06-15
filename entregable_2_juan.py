"""
Commit 2: Se implementaron los modelos de clasificación supervisada y no supervisada
"""

import os
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Configurar backend no interactivo antes de importar pyplot
import matplotlib.pyplot as plt
import seaborn as sns

# Configurar logs de TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import regularizers

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.svm import SVC
from sklearn.cluster import KMeans, DBSCAN as SklearnDBSCAN
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    roc_curve, auc, precision_recall_curve, average_precision_score, silhouette_score
)
from sklearn.decomposition import PCA

try:
    from kneed import KneeLocator
except ImportError:
    KneeLocator = None

# Fijar semillas para reproducibilidad
SEED = 41
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Crear directorio para los gráficos
os.makedirs('plots', exist_ok=True)

# 1. Cargar y muestrear el dataset
print("Cargando y muestreando el dataset...")
df = pd.read_csv('wine_quality_merged.csv')
df = df.sample(n=5150, random_state=SEED).reset_index(drop=True)

# CC: Conversión de características categóricas
df['type'] = df['type'].map({'red': 1, 'white': 0})

# Separar características y variable objetivo
X_raw = df.drop('quality', axis=1)
y_raw = df['quality']

# Mapeo de clases de calidad (3-9) a (0-6) para la red neuronal
class_mapping = {val: idx for idx, val in enumerate(sorted(y_raw.unique()))}
inv_class_mapping = {idx: val for val, idx in class_mapping.items()}
num_classes = len(class_mapping)

print(f"Dataset cargado: {df.shape[0]} muestras, {df.shape[1]} características.")
print("Distribución inicial de clases (quality):", dict(y_raw.value_counts().sort_index()))

# 2. Función de preparación de datos
def prepare_data(escalado, outliers, balanceo):
    """
    Preprocesa los datos según las opciones seleccionadas.
    - escalado: 'si' o 'no'
    - outliers: 'si' (mantener) o 'no' (eliminar 5%)
    - balanceo: 'si' (SMOTE) o 'no'
    """
    X = X_raw.copy()
    y = y_raw.copy()
    
    # 2.1 Manejo de outliers
    if outliers == 'no':
        # IsolationForest para detectar y remover el 5% de outliers
        iso_forest = IsolationForest(contamination=0.05, random_state=SEED)
        outlier_mask = iso_forest.fit_predict(X) == 1
        X = X[outlier_mask].reset_index(drop=True)
        y = y[outlier_mask].reset_index(drop=True)
        
    # 2.2 Escalado de datos
    scaler = None
    if escalado == 'si':
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X = pd.DataFrame(X_scaled, columns=X.columns)
        
    # 2.3 Balanceo de clases (SMOTE con vecinos impares, k_neighbors=3)
    if balanceo == 'si':
        # k_neighbors=3 cumple la sugerencia de usar vecinos impares y funciona para la clase 9 (que tiene 5 muestras)
        smote = SMOTE(random_state=SEED, k_neighbors=3)
        X_res, y_res = smote.fit_resample(X, y)
        X = pd.DataFrame(X_res, columns=X.columns)
        y = pd.Series(y_res)
        
    return X, y, scaler

# Crear las 8 variaciones
variaciones = [
    {'escalado': 'no', 'outliers': 'no', 'balanceo': 'no'},  # Caso 1
    {'escalado': 'no', 'outliers': 'no', 'balanceo': 'si'},  # Caso 2
    {'escalado': 'no', 'outliers': 'si', 'balanceo': 'no'},  # Caso 3
    {'escalado': 'no', 'outliers': 'si', 'balanceo': 'si'},  # Caso 4
    {'escalado': 'si', 'outliers': 'no', 'balanceo': 'no'},  # Caso 5
    {'escalado': 'si', 'outliers': 'no', 'balanceo': 'si'},  # Caso 6
    {'escalado': 'si', 'outliers': 'si', 'balanceo': 'no'},  # Caso 7
    {'escalado': 'si', 'outliers': 'si', 'balanceo': 'si'}   # Caso 8
]

# Estructuras para almacenar métricas de cada modelo y caso
resultados_supervisados = []

# Guardar modelos y datos del Caso 8 para las curvas ROC/PR
caso_8_data = {}

# 3. Entrenamiento y Evaluación Supervisada
print("\n--- INICIANDO ENTRENAMIENTO SUPERVISADO ---")

for idx, var in enumerate(variaciones, 1):
    print(f"\nProcesando Caso {idx}: Escalado={var['escalado'].upper()}, Outliers (Mantener)={var['outliers'].upper()}, Balanceo={var['balanceo'].upper()}")
    X_var, y_var, scaler_var = prepare_data(var['escalado'], var['outliers'], var['balanceo'])
    
    # Dividir en entrenamiento y prueba (80/20) con estratificación para asegurar representación de clases
    # Si alguna clase tiene menos de 2 muestras (puede ocurrir tras remoción de outliers en splits extremos),
    # no estratificamos para evitar errores.
    class_counts = y_var.value_counts()
    if class_counts.min() >= 2:
        X_train, X_test, y_train, y_test = train_test_split(
            X_var, y_var, test_size=0.2, random_state=SEED, stratify=y_var
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X_var, y_var, test_size=0.2, random_state=SEED
        )
        
    # --- 3.1 Árboles de Decisión (Criterio Gini, max_depth=3 para visualización limpia) ---
    dt_model = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=SEED)
    dt_model.fit(X_train, y_train)
    dt_pred = dt_model.predict(X_test)
    dt_probs = dt_model.predict_proba(X_test)
    
    # --- 3.2 KNN (Vecinos impares K=3) ---
    knn_model = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
    knn_model.fit(X_train, y_train)
    knn_pred = knn_model.predict(X_test)
    knn_probs = knn_model.predict_proba(X_test)
    
    # --- 3.3 SVM (Kernel RBF, C=1.0, gamma='auto', probabilidad activa) ---
    svm_model = SVC(kernel='rbf', C=1.0, gamma='auto', probability=True, random_state=SEED)
    svm_model.fit(X_train, y_train)
    svm_pred = svm_model.predict(X_test)
    svm_probs = svm_model.predict_proba(X_test)
    
    # --- 3.4 Red Neuronal (Multicapa en Keras con Regularización y Callbacks) ---
    # Mapear etiquetas de y_train e y_test a indices continuos [0, 6]
    y_train_mapped = y_train.map(class_mapping).to_numpy()
    y_test_mapped = y_test.map(class_mapping).to_numpy()
    
    nn_model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        Dropout(0.2),
        Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])
    
    nn_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=0
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-5,
        verbose=0
    )
    
    # Entrenamiento rápido
    nn_model.fit(
        X_train, y_train_mapped,
        validation_split=0.15,
        epochs=100,
        batch_size=64,
        callbacks=[early_stopping, reduce_lr],
        verbose=0
    )
    
    nn_probs = nn_model.predict(X_test, verbose=0)
    nn_pred_mapped = np.argmax(nn_probs, axis=1)
    # Volver a mapear las predicciones al rango original (3-9)
    nn_pred = np.array([inv_class_mapping[idx] for idx in nn_pred_mapped])
    
    # Guardar datos del Caso 8 para análisis posterior
    if idx == 8:
        caso_8_data = {
            'X_train': X_train, 'y_train': y_train,
            'X_test': X_test, 'y_test': y_test,
            'dt': (dt_model, dt_pred, dt_probs),
            'knn': (knn_model, knn_pred, knn_probs),
            'svm': (svm_model, svm_pred, svm_probs),
            'nn': (nn_model, nn_pred, nn_probs)
        }
        
    # Calcular Métricas (Weighted average para multiclase)
    def calc_metrics(y_true, y_pred):
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        return round(acc, 2), round(prec, 2), round(rec, 2), round(f1, 2)
    
    dt_metrics = calc_metrics(y_test, dt_pred)
    knn_metrics = calc_metrics(y_test, knn_pred)
    svm_metrics = calc_metrics(y_test, svm_pred)
    nn_metrics = calc_metrics(y_test, nn_pred)
    
    # Almacenar en tabla consolidada
    resultados_supervisados.append({
        'Caso': idx,
        'Normalizacion': f"CC(SI), ED({var['escalado'].upper()})",
        'Outliers': var['outliers'].upper(),
        'Balanceo': var['balanceo'].upper(),
        'DT_acc': dt_metrics[0], 'DT_prec': dt_metrics[1], 'DT_rec': dt_metrics[2], 'DT_f1': dt_metrics[3],
        'KNN_acc': knn_metrics[0], 'KNN_prec': knn_metrics[1], 'KNN_rec': knn_metrics[2], 'KNN_f1': knn_metrics[3],
        'SVM_acc': svm_metrics[0], 'SVM_prec': svm_metrics[1], 'SVM_rec': svm_metrics[2], 'SVM_f1': svm_metrics[3],
        'NN_acc': nn_metrics[0], 'NN_prec': nn_metrics[1], 'NN_rec': nn_metrics[2], 'NN_f1': nn_metrics[3],
    })
    
    # Generar 3 casos de prueba para este caso
    print(f"--- 3 Casos de Prueba (Caso {idx}) ---")
    test_indices = list(range(min(3, len(y_test))))
    for t_i in test_indices:
        real_val = y_test.iloc[t_i]
        dt_p = dt_pred[t_i]
        knn_p = knn_pred[t_i]
        svm_p = svm_pred[t_i]
        nn_p = nn_pred[t_i]
        print(f"  Muestra {t_i+1}: Real={real_val} | Pred DT={dt_p}, KNN={knn_p}, SVM={svm_p}, NN={nn_p}")
        
    # Visualizaciones por Caso
    # Decision Tree Importances and Tree structure
    plt.figure(figsize=(8, 4))
    importances = dt_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    sns.barplot(x=importances[indices], y=X_train.columns[indices], palette="viridis")
    plt.title(f"Importancia de Características DT - Caso {idx}")
    plt.tight_layout()
    plt.savefig(f"plots/dt_importance_caso_{idx}.png")
    plt.close()
    
    plt.figure(figsize=(15, 8))
    plot_tree(dt_model, filled=True, feature_names=X_train.columns, 
              class_names=[str(c) for c in dt_model.classes_], rounded=True, fontsize=10)
    plt.title(f"Árbol de Decisión - Caso {idx}")
    plt.tight_layout()
    plt.savefig(f"plots/dt_tree_caso_{idx}.png")
    plt.close()
    
    # Proyecciones y Regiones de decisión
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train)
    
    # Crear malla para regiones de decisión con resolución fija
    x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
    y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    
    # Graficar Regiones de Decisión para KNN (entrenado en PCA para visualización clara)
    knn_pca = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
    knn_pca.fit(X_train_pca, y_train)
    Z_knn = knn_pca.predict(np.c_[xx.ravel(), yy.ravel()])
    Z_knn = Z_knn.reshape(xx.shape)
    
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z_knn, alpha=0.3, cmap='coolwarm')
    sns.scatterplot(x=X_train_pca[:, 0], y=X_train_pca[:, 1], hue=y_train, palette='tab10', edgecolor='k', alpha=0.8)
    plt.title(f"Regiones de Decisión KNN (Proyección PCA) - Caso {idx}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(f"plots/knn_regions_caso_{idx}.png")
    plt.close()
    
    # Graficar Regiones de Decisión para SVM
    svm_pca = SVC(kernel='rbf', C=1.0, gamma='auto', random_state=SEED)
    svm_pca.fit(X_train_pca, y_train)
    Z_svm = svm_pca.predict(np.c_[xx.ravel(), yy.ravel()])
    Z_svm = Z_svm.reshape(xx.shape)
    
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z_svm, alpha=0.3, cmap='coolwarm')
    sns.scatterplot(x=X_train_pca[:, 0], y=X_train_pca[:, 1], hue=y_train, palette='tab10', edgecolor='k', alpha=0.8)
    plt.title(f"Regiones de Decisión SVM (Proyección PCA) - Caso {idx}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(f"plots/svm_regions_caso_{idx}.png")
    plt.close()

# Convertir tabla supervisada a DataFrame
df_supervisados = pd.DataFrame(resultados_supervisados)
print("\n--- FIGURA 1: TABLA COMPARATIVA SUPERVISADA ---")
print(df_supervisados.to_string(index=False))

# Guardar tabla en CSV para reporte
df_supervisados.to_csv('plots/figura_1_supervisado.csv', index=False)

# 4. Gráficos de barra comparativos para F1-Score (Supervisado)
# 4.1 F1-Score Máximo General por Algoritmo
f1_max_dt = df_supervisados['DT_f1'].max()
f1_max_knn = df_supervisados['KNN_f1'].max()
f1_max_svm = df_supervisados['SVM_f1'].max()
f1_max_nn = df_supervisados['NN_f1'].max()

plt.figure(figsize=(7, 5))
algorithms = ['Decision Tree', 'KNN', 'SVM', 'Neural Network']
max_f1_scores = [f1_max_dt, f1_max_knn, f1_max_svm, f1_max_nn]
sns.barplot(x=algorithms, y=max_f1_scores, palette="muted")
plt.ylabel("F1-Score Máximo")
plt.title("F1-Score Máximo por Algoritmo (General)")
plt.ylim(0, 1.05)
for i, val in enumerate(max_f1_scores):
    plt.text(i, val + 0.01, f"{val:.2f}", ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.savefig('plots/supervised_f1_max.png')
plt.close()

# Helper para calcular y graficar F1-Score promedio por agrupación de preprocesamiento
def plot_grouped_f1(group_col, group_title, save_name):
    # Reestructurar datos para facilitar la agrupación
    data_list = []
    for _, row in df_supervisados.iterrows():
        # Obtener los flags de preprocesamiento
        is_ed_si = "ED(SI)" in row['Normalizacion']
        is_outliers_si = row['Outliers'] == "SI"
        is_balanceo_si = row['Balanceo'] == "SI"
        
        # Determinar el valor de la agrupación
        if group_col == 'Normalizacion':
            val = "Con Escalado" if is_ed_si else "Sin Escalado"
        elif group_col == 'Outliers':
            val = "Con Outliers (5%)" if is_outliers_si else "Sin Outliers"
        elif group_col == 'Balanceo':
            val = "Con Balanceo" if is_balanceo_si else "Sin Balanceo"
            
        data_list.append({'Group': val, 'Algoritmo': 'Decision Tree', 'F1': row['DT_f1']})
        data_list.append({'Group': val, 'Algoritmo': 'KNN', 'F1': row['KNN_f1']})
        data_list.append({'Group': val, 'Algoritmo': 'SVM', 'F1': row['SVM_f1']})
        data_list.append({'Group': val, 'Algoritmo': 'Neural Network', 'F1': row['NN_f1']})
        
    df_grouped = pd.DataFrame(data_list)
    df_mean = df_grouped.groupby(['Group', 'Algoritmo'])['F1'].mean().reset_index()
    
    plt.figure(figsize=(9, 5))
    sns.barplot(data=df_mean, x='Group', y='F1', hue='Algoritmo', palette='Set2')
    plt.ylabel("F1-Score Promedio")
    plt.xlabel(group_title)
    plt.title(f"F1-Score Medio por {group_title} y Algoritmo")
    plt.ylim(0, 1.05)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'plots/{save_name}.png')
    plt.close()

plot_grouped_f1('Normalizacion', 'Normalización (Escalado)', 'supervised_f1_by_normalization')
plot_grouped_f1('Outliers', 'Manejo de Outliers', 'supervised_f1_by_outliers')
plot_grouped_f1('Balanceo', 'Balanceo de Clases', 'supervised_f1_by_balance')

# 5. Curvas ROC & Precision-Recall para el Caso 8
# Dado que es un problema multiclase con 7 clases, calculamos curvas ROC y PR macro-promediadas (One-vs-Rest)
print("\nGenerando curvas ROC & Precision-Recall para el Caso 8...")
y_test_c8 = caso_8_data['y_test']
y_test_c8_mapped = y_test_c8.map(class_mapping).to_numpy()

# Binarizar y_test para evaluación One-vs-Rest
from sklearn.preprocessing import label_binarize
y_test_bin = label_binarize(y_test_c8_mapped, classes=range(num_classes))

# Modelos y sus probabilidades en Caso 8
probs_dict = {
    'Decision Tree': caso_8_data['dt'][2],
    'KNN': caso_8_data['knn'][2],
    'SVM': caso_8_data['svm'][2],
    'Neural Network': caso_8_data['nn'][2]
}

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

colors = ['darkorange', 'blue', 'green', 'purple']

for (name, probs), color in zip(probs_dict.items(), colors):
    # ROC Macro
    fpr_grid = np.linspace(0.0, 1.0, 100)
    mean_tpr = np.zeros_like(fpr_grid)
    for c_i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, c_i], probs[:, c_i])
        mean_tpr += np.interp(fpr_grid, fpr, tpr)
    mean_tpr /= num_classes
    roc_auc = auc(fpr_grid, mean_tpr)
    axes[0].plot(fpr_grid, mean_tpr, label=f'{name} (AUC = {roc_auc:.2f})', color=color, lw=2)
    
    # PR Macro
    precision_grid = np.linspace(0.0, 1.0, 100)
    mean_precision = np.zeros_like(precision_grid)
    for c_i in range(num_classes):
        prec, rec, _ = precision_recall_curve(y_test_bin[:, c_i], probs[:, c_i])
        # Invertir recall y precisión para interpolar
        mean_precision += np.interp(precision_grid, rec[::-1], prec[::-1])
    mean_precision /= num_classes
    avg_prec = average_precision_score(y_test_bin, probs, average='macro')
    axes[1].plot(precision_grid, mean_precision, label=f'{name} (AP = {avg_prec:.2f})', color=color, lw=2)

# Configurar gráfico ROC
axes[0].plot([0, 1], [0, 1], 'k--', lw=2)
axes[0].set_xlim([0.0, 1.0])
axes[0].set_ylim([0.0, 1.05])
axes[0].set_xlabel('Tasa de Falsos Positivos (FPR)')
axes[0].set_ylabel('Tasa de Verdaderos Positivos (TPR)')
axes[0].set_title('Curva ROC Macro-Promediada (Caso 8)')
axes[0].legend(loc="lower right")
axes[0].grid(alpha=0.3)

# Configurar gráfico PR
axes[1].set_xlim([0.0, 1.0])
axes[1].set_ylim([0.0, 1.05])
axes[1].set_xlabel('Recall')
axes[1].set_ylabel('Precision')
axes[1].set_title('Curva Precision-Recall Macro-Promediada (Caso 8)')
axes[1].legend(loc="lower left")
axes[1].grid(alpha=0.3)

plt.suptitle("Evaluación de Modelos Supervisados en Caso 8")
plt.tight_layout()
plt.savefig('plots/supervised_case8_roc_pr.png')
plt.close()


# 6. APRENDIZAJE NO SUPERVISADO
print("\n--- INICIANDO APRENDIZAJE NO SUPERVISADO ---")

# Almacenar métricas de clustering
resultados_no_supervisados = []
unsupervised_cases_data = []

for idx, var in enumerate(variaciones, 1):
    print(f"\nProcesando Caso No Supervisado {idx}: Escalado={var['escalado'].upper()}, Outliers (Mantener)={var['outliers'].upper()}")
    
    # Preparar datos sin la variable objetivo
    X_var, y_var, _ = prepare_data(var['escalado'], var['outliers'], balanceo='no')
    
    # --- 6.1 KMeans: Método del codo y Silhouette ---
    inertias = []
    sil_scores = []
    k_range = range(2, 11)
    
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
        labels = km.fit_predict(X_var)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X_var, labels))
        
    # Determinar K óptimo
    # Método 1: Silhouette máximo
    best_k_sil = k_range[np.argmax(sil_scores)]
    best_sil = max(sil_scores)
    
    # Método 2: Codo (KneeLocator)
    best_k_elbow = 3  # fallback estándar
    if KneeLocator is not None:
        kl = KneeLocator(list(k_range), inertias, curve='convex', direction='decreasing')
        if kl.elbow is not None:
            best_k_elbow = kl.elbow
            
    # Elegimos K basado en silhouette máximo para optimizar separación de clústeres
    opt_k = best_k_sil
    km_final = KMeans(n_clusters=opt_k, random_state=SEED, n_init=10)
    km_labels = km_final.fit_predict(X_var)
    km_inertia = km_final.inertia_
    km_sil = silhouette_score(X_var, km_labels)
    
    # Guardar gráfico de codo y silhouette
    fig, ax1 = plt.subplots(figsize=(8, 4))
    color = 'tab:red'
    ax1.set_xlabel('Número de Clústeres (K)')
    ax1.set_ylabel('Inercia', color=color)
    ax1.plot(k_range, inertias, 'o-', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Silhouette Score', color=color)
    ax2.plot(k_range, sil_scores, 's--', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title(f"Método del Codo y Silhouette para KMeans - Caso {idx} (K Óptimo = {opt_k})")
    fig.tight_layout()
    plt.savefig(f"plots/kmeans_elbow_silhouette_caso_{idx}.png")
    plt.close()
    
    # --- 6.2 DBSCAN: Ajuste de MinPts y Épsilon ---
    # MinPts = 2 * n_features
    min_pts = 2 * X_var.shape[1]
    
    # Calcular distancias de NearestNeighbors para gráfico de K-distancia
    neighbors = NearestNeighbors(n_neighbors=min_pts)
    neighbors.fit(X_var)
    distances, _ = neighbors.kneighbors(X_var)
    k_distances = np.sort(distances[:, -1])
    
    # Encontrar codo del gráfico de K-distancia
    eps_opt = 0.5  # fallback
    if KneeLocator is not None:
        kl_eps = KneeLocator(range(len(k_distances)), k_distances, curve='convex', direction='increasing')
        if kl_eps.knee is not None:
            eps_opt = k_distances[kl_eps.knee]
    else:
        eps_opt = np.percentile(k_distances, 90)
        
    # Ajuste robusto de Épsilon: si el eps óptimo da colapso de clústeres, buscar candidatos cercanos
    eps_candidates = [eps_opt * factor for factor in [0.75, 0.9, 1.0, 1.1, 1.25]]
    best_dbscan_sil = -2.0
    best_dbscan_labels = None
    best_dbscan_eps = eps_opt
    best_dbscan_model = None
    
    for eps_cand in eps_candidates:
        dbs = SklearnDBSCAN(eps=eps_cand, min_samples=min_pts)
        dbs_labels = dbs.fit_predict(X_var)
        unique_labels = set(dbs_labels) - {-1}
        
        # Debe haber al menos 2 clústeres (excluyendo ruido) para calcular silhouette
        if len(unique_labels) >= 2:
            # Calcular silhouette solo en puntos no-ruido
            non_noise_mask = dbs_labels != -1
            if np.sum(non_noise_mask) > len(unique_labels):
                score = silhouette_score(X_var[non_noise_mask], dbs_labels[non_noise_mask])
                if score > best_dbscan_sil:
                    best_dbscan_sil = score
                    best_dbscan_labels = dbs_labels
                    best_dbscan_eps = eps_cand
                    best_dbscan_model = dbs
                    
    # Si ningún candidato fue válido (colapso de clústeres), usamos el óptimo del codo original
    if best_dbscan_labels is None:
        best_dbscan_model = SklearnDBSCAN(eps=eps_opt, min_samples=min_pts)
        best_dbscan_labels = best_dbscan_model.fit_predict(X_var)
        non_noise_mask = best_dbscan_labels != -1
        unique_labels = set(best_dbscan_labels) - {-1}
        if len(unique_labels) >= 2 and np.sum(non_noise_mask) > len(unique_labels):
            best_dbscan_sil = silhouette_score(X_var[non_noise_mask], best_dbscan_labels[non_noise_mask])
        else:
            best_dbscan_sil = -1.0 # Indicar mala separación o colapso
            
    # Guardar gráfico de K-distancia
    plt.figure(figsize=(8, 4))
    plt.plot(k_distances, color='blue', lw=2)
    plt.axhline(best_dbscan_eps, color='red', linestyle='--', label=f'Eps Elegido = {best_dbscan_eps:.3f}')
    plt.title(f"Gráfico de K-distancia para DBSCAN (MinPts = {min_pts}) - Caso {idx}")
    plt.xlabel("Puntos ordenados por distancia")
    plt.ylabel(f"Distancia al vecino {min_pts}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"plots/dbscan_k_distance_caso_{idx}.png")
    plt.close()
    
    # Registrar métricas consolidado no supervisado
    resultados_no_supervisados.append({
        'Caso': idx,
        'Normalizacion': f"CC(SI), ED({var['escalado'].upper()})",
        'Outliers': var['outliers'].upper(),
        'Balanceo': "NO",  # No supervisado típicamente no lleva SMOTE en la variable objetivo
        'KMeans_Inertia': round(km_inertia, 2),
        'KMeans_Silhouette': round(km_sil, 3),
        'DBSCAN_Silhouette': round(best_dbscan_sil, 3)
    })
    
    # 3 Casos de prueba para no supervisado: mostrar clúster asignado
    print(f"--- 3 Casos de Prueba Clustering (Caso {idx}) ---")
    test_indices = [0, 1, 2]
    for t_i in test_indices:
        km_c = km_labels[t_i]
        dbs_c = best_dbscan_labels[t_i]
        dbs_label = f"Clúster {dbs_c}" if dbs_c != -1 else "Ruido / Outlier"
        
        # Calcular distancias a los centroides para KMeans
        raw_row = X_var.iloc[[t_i]]
        km_dist = km_final.transform(raw_row)[0].round(3).tolist()
        
        print(f"  Muestra {t_i+1}: KMeans Clúster={km_c} (Dists a centroides={km_dist}) | DBSCAN={dbs_label}")
        
    # Visualización de Clústeres en 2D usando PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_var)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Gráfico KMeans
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=km_labels, palette='tab10', ax=axes[0], s=15, alpha=0.7)
    # Graficar centroides proyectados
    centroids_pca = pca.transform(km_final.cluster_centers_)
    axes[0].scatter(centroids_pca[:, 0], centroids_pca[:, 1], c='black', marker='X', s=150, edgecolors='white', label='Centroides')
    axes[0].set_title(f"K-Means (K = {opt_k}) - Caso {idx}")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Gráfico DBSCAN
    dbs_hue = [f"Clúster {l}" if l != -1 else "Ruido" for l in best_dbscan_labels]
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=dbs_hue, palette='tab10', ax=axes[1], s=15, alpha=0.7)
    axes[1].set_title(f"DBSCAN (Eps = {best_dbscan_eps:.3f}, MinPts = {min_pts}) - Caso {idx}")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.suptitle(f"Proyección PCA de Agrupamiento - Caso {idx}")
    plt.tight_layout()
    plt.savefig(f"plots/clustering_pca_caso_{idx}.png")
    plt.close()
    
    # Guardar tablas cruzadas para el análisis del dominio del problema
    unsupervised_cases_data.append({
        'idx': idx,
        'X': X_var,
        'y_quality': y_var,
        'y_type': df.loc[X_var.index, 'type'],
        'km_labels': km_labels,
        'dbs_labels': best_dbscan_labels
    })

# Convertir tabla no supervisada a DataFrame
df_no_supervisados = pd.DataFrame(resultados_no_supervisados)
print("\n--- FIGURA 3: TABLA COMPARATIVA NO SUPERVISADA ---")
print(df_no_supervisados.to_string(index=False))

# Guardar tabla no supervisada en CSV
df_no_supervisados.to_csv('plots/figura_3_no_supervisado.csv', index=False)

# 7. Gráfico de barras comparativo de Silhouette_Score
plt.figure(figsize=(10, 5))
labels = [f'Caso {i}' for i in range(1, 9)]
x = np.arange(len(labels))
width = 0.35

plt.bar(x - width/2, df_no_supervisados['KMeans_Silhouette'], width, label='K-Means', color='steelblue')
plt.bar(x + width/2, df_no_supervisados['DBSCAN_Silhouette'], width, label='DBSCAN', color='mediumpurple')

plt.ylabel('Silhouette Score')
plt.title('Silhouette Score por Algoritmo y Caso')
plt.xticks(x, labels)
plt.ylim(-0.2, 1.0)
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('plots/unsupervised_silhouette_scores.png')
plt.close()

# 8. Guardar reportes de Tablas Cruzadas para análisis en el documento final
with open('plots/tablas_cruzadas_clustering.txt', 'w') as f:
    for data in unsupervised_cases_data:
        f.write(f"\n======================================================\n")
        f.write(f"ANÁLISIS DE DOMINIO - CASO {data['idx']}\n")
        f.write(f"======================================================\n")
        
        # Cruzar Clústeres KMeans con Calidad del Vino
        km_quality = pd.crosstab(data['km_labels'], data['y_quality'])
        f.write("\nK-Means Clústeres vs Calidad (Real):\n")
        f.write(km_quality.to_string())
        f.write("\n")
        
        # Cruzar Clústeres KMeans con Tipo de Vino (Red=1, White=0)
        type_labels = data['y_type'].map({1: 'Red', 0: 'White'})
        km_type = pd.crosstab(data['km_labels'], type_labels)
        f.write("\nK-Means Clústeres vs Tipo de Vino:\n")
        f.write(km_type.to_string())
        f.write("\n")
        
        # Cruzar Clústeres DBSCAN con Calidad del Vino
        dbs_quality = pd.crosstab(data['dbs_labels'], data['y_quality'])
        f.write("\nDBSCAN Clústeres (Ruido=-1) vs Calidad (Real):\n")
        f.write(dbs_quality.to_string())
        f.write("\n")
        
        # Cruzar Clústeres DBSCAN con Tipo de Vino
        dbs_type = pd.crosstab(data['dbs_labels'], type_labels)
        f.write("\nDBSCAN Clústeres (Ruido=-1) vs Tipo de Vino:\n")
        f.write(dbs_type.to_string())
        f.write("\n")

print("\n¡Ejecución completada! Todos los gráficos se guardaron en la carpeta 'plots/'.")
print("Se ha generado el archivo 'plots/tablas_cruzadas_clustering.txt' con el análisis de dominio.")
