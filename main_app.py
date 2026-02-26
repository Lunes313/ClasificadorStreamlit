"""
Clasificador IRIS - Aplicación Streamlit
Clasificador dinámico y pedagógico para el dataset IRIS con visualizaciones
de desempeño e interfaz de predicción.
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# ──────────────────────────────────────────────
# Configuración de página
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Clasificador IRIS",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Descripción pedagógica de cada clasificador
# ──────────────────────────────────────────────
DESCRIPCIONES = {
    "K-Vecinos Más Cercanos (KNN)": (
        "**K-Nearest Neighbors** clasifica cada punto nuevo según las *k* muestras de "
        "entrenamiento más cercanas. Es un algoritmo **no paramétrico** y **lazy** "
        "(no construye un modelo explícito). La distancia Euclidiana es la métrica más común. "
        "Aumentar *k* reduce el sobreajuste pero puede incrementar el sesgo."
    ),
    "Árbol de Decisión": (
        "Un **Árbol de Decisión** divide el espacio de características con reglas "
        "if/else que maximizan la pureza de los nodos (Gini o Entropía). Es muy "
        "interpretable y puede sobreajustarse si se deja crecer sin límites. "
        "La *profundidad máxima* controla la complejidad del modelo."
    ),
    "Bosque Aleatorio": (
        "**Random Forest** construye múltiples árboles de decisión sobre submuestras "
        "aleatorias del dataset y promedia sus predicciones (**bagging**). Reduce el "
        "sobreajuste de los árboles individuales y es robusto ante ruido. "
        "El número de estimadores y la profundidad máxima son los hiperparámetros clave."
    ),
    "Máquina de Soporte Vectorial (SVM)": (
        "Una **SVM** busca el hiperplano de máximo margen que separa las clases. "
        "Con el *kernel RBF* puede capturar fronteras no lineales proyectando los datos "
        "a un espacio de mayor dimensión. El parámetro *C* regula el trade-off entre "
        "margen amplio y errores de clasificación."
    ),
    "Regresión Logística": (
        "La **Regresión Logística** modela la probabilidad de pertenencia a cada clase "
        "mediante una función sigmoide o softmax (multiclase). Es un clasificador lineal "
        "rápido e interpretable. El parámetro *C* controla la regularización (mayor C = "
        "menos regularización)."
    ),
}

# ──────────────────────────────────────────────
# Carga del dataset
# ──────────────────────────────────────────────
@st.cache_data
def cargar_datos():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["especie"] = pd.Categorical.from_codes(iris.target, iris.target_names)
    return df, iris


# ──────────────────────────────────────────────
# Construcción del clasificador según selección
# ──────────────────────────────────────────────
def crear_clasificador(nombre, params):
    if nombre == "K-Vecinos Más Cercanos (KNN)":
        return KNeighborsClassifier(n_neighbors=params["k"])
    if nombre == "Árbol de Decisión":
        return DecisionTreeClassifier(
            max_depth=params["max_depth"] if params["max_depth"] > 0 else None,
            criterion=params["criterion"],
            random_state=42,
        )
    if nombre == "Bosque Aleatorio":
        return RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"] if params["max_depth"] > 0 else None,
            random_state=42,
        )
    if nombre == "Máquina de Soporte Vectorial (SVM)":
        return SVC(C=params["C"], kernel="rbf", probability=True, random_state=42)
    # Regresión Logística
    return LogisticRegression(C=params["C"], max_iter=5000, random_state=42)


# ──────────────────────────────────────────────
# Gráfica de distribución por especie
# ──────────────────────────────────────────────
def graf_distribucion(df):
    fig = px.histogram(
        df,
        x="especie",
        color="especie",
        title="Distribución de muestras por especie",
        labels={"especie": "Especie", "count": "Cantidad"},
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_layout(showlegend=False)
    return fig


# ──────────────────────────────────────────────
# Gráfica de pares (scatter matrix)
# ──────────────────────────────────────────────
def graf_pares(df):
    fig = px.scatter_matrix(
        df,
        dimensions=df.columns[:4],
        color="especie",
        title="Matriz de dispersión de características",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_traces(diagonal_visible=False, marker=dict(size=4))
    return fig


# ──────────────────────────────────────────────
# Gráfica de correlación (heatmap)
# ──────────────────────────────────────────────
def graf_correlacion(df):
    corr = df.iloc[:, :4].corr()
    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        title="Mapa de correlación entre características",
        aspect="auto",
    )
    return fig


# ──────────────────────────────────────────────
# Matriz de confusión
# ──────────────────────────────────────────────
def graf_confusion(y_test, y_pred, clases):
    cm = confusion_matrix(y_test, y_pred)
    fig = px.imshow(
        cm,
        x=clases,
        y=clases,
        text_auto=True,
        color_continuous_scale="Blues",
        title="Matriz de Confusión",
        labels=dict(x="Predicción", y="Real", color="Cantidad"),
    )
    fig.update_layout(xaxis_title="Predicción", yaxis_title="Real")
    return fig


# ──────────────────────────────────────────────
# Curvas ROC multiclase (One-vs-Rest)
# ──────────────────────────────────────────────
def graf_roc(modelo, X_test, y_test, clases):
    y_bin = label_binarize(y_test, classes=list(range(len(clases))))
    try:
        y_score = modelo.predict_proba(X_test)
    except AttributeError:
        return None

    fig = go.Figure()
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for i, cls in enumerate(clases):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=f"{cls} (AUC = {roc_auc:.2f})",
                line=dict(color=colors[i]),
            )
        )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Azar",
            line=dict(color="gray", dash="dash"),
        )
    )
    fig.update_layout(
        title="Curvas ROC (One-vs-Rest)",
        xaxis_title="Tasa de Falsos Positivos",
        yaxis_title="Tasa de Verdaderos Positivos",
        legend_title="Clase",
    )
    return fig


# ──────────────────────────────────────────────
# Importancia de características (si aplica)
# ──────────────────────────────────────────────
def graf_importancia(modelo, feature_names, nombre_clf):
    importancias = None
    if hasattr(modelo, "feature_importances_"):
        importancias = modelo.feature_importances_
    elif hasattr(modelo, "coef_"):
        importancias = np.abs(modelo.coef_).mean(axis=0)

    if importancias is None:
        return None

    df_imp = pd.DataFrame(
        {"Característica": feature_names, "Importancia": importancias}
    ).sort_values("Importancia", ascending=True)

    fig = px.bar(
        df_imp,
        x="Importancia",
        y="Característica",
        orientation="h",
        title=f"Importancia de características — {nombre_clf}",
        color="Importancia",
        color_continuous_scale="Viridis",
    )
    fig.update_layout(coloraxis_showscale=False)
    return fig


# ──────────────────────────────────────────────
# Validación cruzada
# ──────────────────────────────────────────────
def graf_cv(scores):
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=[f"Fold {i+1}" for i in range(len(scores))],
            y=scores,
            marker_color="steelblue",
            text=[f"{s:.3f}" for s in scores],
            textposition="outside",
        )
    )
    fig.add_hline(
        y=scores.mean(),
        line_dash="dash",
        line_color="red",
        annotation_text=f"Media: {scores.mean():.3f}",
    )
    fig.update_layout(
        title="Validación Cruzada (5-Fold)",
        xaxis_title="Fold",
        yaxis_title="Exactitud",
        yaxis=dict(range=[0, 1.05]),
    )
    return fig


# ══════════════════════════════════════════════
#                  APP PRINCIPAL
# ══════════════════════════════════════════════
def main():
    st.title("🌸 Clasificador IRIS — Streamlit")
    st.markdown(
        "Aplicación **pedagógica e interactiva** para explorar algoritmos de "
        "clasificación sobre el clásico dataset *Iris* de Fisher (1936)."
    )

    df, iris = cargar_datos()
    feature_names = iris.feature_names
    target_names = list(iris.target_names)

    # ──────────────────────
    # Barra lateral
    # ──────────────────────
    with st.sidebar:
        st.header("⚙️ Configuración")

        clasificador_nombre = st.selectbox(
            "Algoritmo de clasificación",
            list(DESCRIPCIONES.keys()),
        )

        st.subheader("Hiperparámetros")
        params = {}
        if clasificador_nombre == "K-Vecinos Más Cercanos (KNN)":
            params["k"] = st.slider("Número de vecinos (k)", 1, 20, 5)
        elif clasificador_nombre == "Árbol de Decisión":
            params["max_depth"] = st.slider(
                "Profundidad máxima (0 = sin límite)", 0, 15, 4
            )
            params["criterion"] = st.selectbox("Criterio", ["gini", "entropy"])
        elif clasificador_nombre == "Bosque Aleatorio":
            params["n_estimators"] = st.slider("Número de árboles", 10, 300, 100)
            params["max_depth"] = st.slider(
                "Profundidad máxima (0 = sin límite)", 0, 15, 0
            )
        elif clasificador_nombre in (
            "Máquina de Soporte Vectorial (SVM)",
            "Regresión Logística",
        ):
            params["C"] = st.slider("Parámetro C (regularización)", 0.01, 10.0, 1.0)

        st.subheader("Partición de datos")
        test_size = st.slider("Tamaño del conjunto de prueba (%)", 10, 40, 20) / 100
        escalar = st.checkbox("Escalar características (StandardScaler)", value=True)

    # ──────────────────────
    # Pestañas
    # ──────────────────────
    tab_datos, tab_modelo, tab_prediccion = st.tabs(
        ["📊 Exploración de Datos", "🤖 Modelo y Desempeño", "🔮 Predicción"]
    )

    # ══════════════════════
    # PESTAÑA 1: DATOS
    # ══════════════════════
    with tab_datos:
        st.header("Dataset IRIS")
        col1, col2, col3 = st.columns(3)
        col1.metric("Muestras totales", len(df))
        col2.metric("Características", len(feature_names))
        col3.metric("Clases", len(target_names))

        with st.expander("📋 Ver primeras filas del dataset"):
            st.dataframe(df.head(10), use_container_width=True)

        with st.expander("📈 Estadísticas descriptivas"):
            st.dataframe(df.describe(), use_container_width=True)

        st.subheader("Visualizaciones exploratorias")
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(graf_distribucion(df), use_container_width=True)
        with c2:
            st.plotly_chart(graf_correlacion(df), use_container_width=True)

        st.plotly_chart(graf_pares(df), use_container_width=True)

    # ══════════════════════
    # PESTAÑA 2: MODELO
    # ══════════════════════
    with tab_modelo:
        st.header(f"Algoritmo: {clasificador_nombre}")

        # Explicación pedagógica
        with st.expander("📚 ¿Cómo funciona este algoritmo?", expanded=True):
            st.markdown(DESCRIPCIONES[clasificador_nombre])

        # Entrenamiento
        X = df[feature_names].values
        y = iris.target

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        if escalar:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        modelo = crear_clasificador(clasificador_nombre, params)
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)

        acc_train = accuracy_score(y_train, modelo.predict(X_train))
        acc_test = accuracy_score(y_test, y_pred)

        # Métricas principales
        st.subheader("📏 Métricas de Desempeño")
        m1, m2, m3 = st.columns(3)
        m1.metric("Exactitud en entrenamiento", f"{acc_train:.2%}")
        m2.metric("Exactitud en prueba", f"{acc_test:.2%}")
        delta = acc_test - acc_train
        m3.metric(
            "Diferencia (sobreajuste)",
            f"{delta:.2%}",
            delta=f"{delta:.2%}",
            delta_color="inverse",
        )

        # Validación cruzada
        st.subheader("🔄 Validación Cruzada (5-Fold)")
        clf_cv = crear_clasificador(clasificador_nombre, params)
        if escalar:
            cv_pipeline = Pipeline([("scaler", StandardScaler()), ("clf", clf_cv)])
        else:
            cv_pipeline = clf_cv
        scores_cv = cross_val_score(cv_pipeline, X, y, cv=5)
        st.plotly_chart(graf_cv(scores_cv), use_container_width=True)

        # Gráficas de desempeño
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("🗂️ Matriz de Confusión")
            st.plotly_chart(
                graf_confusion(y_test, y_pred, target_names), use_container_width=True
            )
        with col_b:
            st.subheader("📈 Curvas ROC")
            fig_roc = graf_roc(modelo, X_test, y_test, target_names)
            if fig_roc:
                st.plotly_chart(fig_roc, use_container_width=True)
            else:
                st.info("Este clasificador no soporta probabilidades para ROC.")

        # Reporte de clasificación
        st.subheader("📄 Reporte de Clasificación")
        report_dict = classification_report(
            y_test, y_pred, target_names=target_names, output_dict=True
        )
        df_report = pd.DataFrame(report_dict).transpose()
        st.dataframe(df_report.style.format("{:.2f}"), use_container_width=True)

        # Importancia de características
        fig_imp = graf_importancia(modelo, feature_names, clasificador_nombre)
        if fig_imp:
            st.subheader("🔍 Importancia de Características")
            st.plotly_chart(fig_imp, use_container_width=True)

        # Árbol de decisión: visualización del árbol
        if clasificador_nombre == "Árbol de Decisión":
            with st.expander("🌳 Ver estructura del árbol de decisión"):
                fig_tree, ax = plt.subplots(figsize=(16, 6))
                plot_tree(
                    modelo,
                    feature_names=feature_names,
                    class_names=target_names,
                    filled=True,
                    rounded=True,
                    fontsize=9,
                    ax=ax,
                )
                st.pyplot(fig_tree)
                plt.close(fig_tree)

                rules = export_text(
                    modelo, feature_names=list(feature_names)
                )
                st.code(rules, language="text")

    # ══════════════════════
    # PESTAÑA 3: PREDICCIÓN
    # ══════════════════════
    with tab_prediccion:
        st.header("🔮 Interfaz de Predicción")
        st.markdown(
            "Introduce los valores de las características para obtener la predicción "
            "del clasificador entrenado y la probabilidad estimada para cada clase."
        )

        # Reentrenar con todos los datos para predecir
        X_full = df[feature_names].values
        if escalar:
            scaler_full = StandardScaler()
            X_full_scaled = scaler_full.fit_transform(X_full)
        else:
            X_full_scaled = X_full
            scaler_full = None

        modelo_full = crear_clasificador(clasificador_nombre, params)
        modelo_full.fit(X_full_scaled, iris.target)

        st.subheader("Introduce las medidas de la flor")
        stats = df[feature_names].describe()

        col1, col2 = st.columns(2)
        with col1:
            sepal_length = st.number_input(
                feature_names[0],
                min_value=float(stats.loc["min", feature_names[0]]),
                max_value=float(stats.loc["max", feature_names[0]]),
                value=float(stats.loc["mean", feature_names[0]]),
                step=0.1,
                format="%.1f",
            )
            sepal_width = st.number_input(
                feature_names[1],
                min_value=float(stats.loc["min", feature_names[1]]),
                max_value=float(stats.loc["max", feature_names[1]]),
                value=float(stats.loc["mean", feature_names[1]]),
                step=0.1,
                format="%.1f",
            )
        with col2:
            petal_length = st.number_input(
                feature_names[2],
                min_value=float(stats.loc["min", feature_names[2]]),
                max_value=float(stats.loc["max", feature_names[2]]),
                value=float(stats.loc["mean", feature_names[2]]),
                step=0.1,
                format="%.1f",
            )
            petal_width = st.number_input(
                feature_names[3],
                min_value=float(stats.loc["min", feature_names[3]]),
                max_value=float(stats.loc["max", feature_names[3]]),
                value=float(stats.loc["mean", feature_names[3]]),
                step=0.1,
                format="%.1f",
            )

        if st.button("🌸 Predecir especie", type="primary"):
            entrada = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
            if scaler_full:
                entrada_scaled = scaler_full.transform(entrada)
            else:
                entrada_scaled = entrada

            pred_idx = modelo_full.predict(entrada_scaled)[0]
            pred_clase = target_names[pred_idx]

            st.success(f"### Especie predicha: **{pred_clase}** 🌸")

            if hasattr(modelo_full, "predict_proba"):
                probas = modelo_full.predict_proba(entrada_scaled)[0]
                df_prob = pd.DataFrame(
                    {"Especie": target_names, "Probabilidad": probas}
                )
                fig_prob = px.bar(
                    df_prob,
                    x="Especie",
                    y="Probabilidad",
                    color="Especie",
                    title="Probabilidad estimada por clase",
                    color_discrete_sequence=px.colors.qualitative.Set2,
                    text=[f"{p:.1%}" for p in probas],
                )
                fig_prob.update_traces(textposition="outside")
                fig_prob.update_layout(
                    yaxis=dict(range=[0, 1.1]),
                    showlegend=False,
                )
                st.plotly_chart(fig_prob, use_container_width=True)

            # Mostrar posición de la muestra en el espacio de características
            st.subheader("📍 Posición de la muestra en el dataset")
            df_plot = df.copy()
            fig_pos = px.scatter(
                df_plot,
                x=feature_names[2],
                y=feature_names[3],
                color="especie",
                title="Muestra ingresada vs dataset (pétalo largo vs ancho)",
                color_discrete_sequence=px.colors.qualitative.Set2,
                opacity=0.6,
            )
            fig_pos.add_trace(
                go.Scatter(
                    x=[petal_length],
                    y=[petal_width],
                    mode="markers",
                    marker=dict(size=16, color="red", symbol="star"),
                    name=f"Tu muestra → {pred_clase}",
                )
            )
            st.plotly_chart(fig_pos, use_container_width=True)


if __name__ == "__main__":
    main()
