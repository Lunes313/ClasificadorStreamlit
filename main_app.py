"""
Clasificador IRIS — Aplicación Streamlit pedagógica e interactiva.

Secciones:
1. Introducción al dataset IRIS
2. Exploración de datos (EDA)
3. Configuración y entrenamiento del clasificador
4. Métricas y gráficas de desempeño
5. Predicción interactiva
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    classification_report,
    confusion_matrix,
    roc_curve,
)
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import label_binarize
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

warnings.filterwarnings("ignore")

# ── Configuración de página ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Clasificador IRIS 🌸",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constantes ───────────────────────────────────────────────────────────────
CLASSIFIERS = {
    "K-Nearest Neighbors (KNN)": "knn",
    "Árbol de Decisión": "tree",
    "Random Forest": "rf",
    "Máquina de Vectores de Soporte (SVM)": "svm",
    "Regresión Logística": "lr",
}

CLASS_NAMES = ["Setosa", "Versicolor", "Virginica"]
FEATURE_NAMES = [
    "Largo sépalo (cm)",
    "Ancho sépalo (cm)",
    "Largo pétalo (cm)",
    "Ancho pétalo (cm)",
]
FEATURE_KEYS = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

PALETTE = ["#4C72B0", "#DD8452", "#55A868"]


# ── Utilidades ────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=FEATURE_NAMES)
    df["Especie"] = [CLASS_NAMES[t] for t in iris.target]
    df["target"] = iris.target
    return df, iris


def build_classifier(name: str, params: dict):
    if name == "knn":
        return KNeighborsClassifier(**params)
    if name == "tree":
        return DecisionTreeClassifier(**params)
    if name == "rf":
        return RandomForestClassifier(**params)
    if name == "svm":
        return SVC(**params, probability=True)
    if name == "lr":
        return LogisticRegression(**params, max_iter=500)
    raise ValueError(f"Clasificador desconocido: {name}")


# ── Sidebar ──────────────────────────────────────────────────────────────────
def sidebar():
    st.sidebar.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/4/41/Iris_versicolor_3.jpg/320px-Iris_versicolor_3.jpg",
        use_container_width=True,
    )
    st.sidebar.title("⚙️ Configuración")

    st.sidebar.header("1. División de datos")
    test_size = st.sidebar.slider(
        "Tamaño del conjunto de prueba (%)", 10, 40, 20, step=5
    )
    random_state = st.sidebar.number_input(
        "Semilla aleatoria (random_state)", 0, 999, 42, step=1
    )

    st.sidebar.header("2. Clasificador")
    clf_label = st.sidebar.selectbox("Algoritmo", list(CLASSIFIERS.keys()))
    clf_key = CLASSIFIERS[clf_label]

    st.sidebar.header("3. Hiperparámetros")
    params = {}
    if clf_key == "knn":
        params["n_neighbors"] = st.sidebar.slider("Vecinos (k)", 1, 20, 5)
        params["weights"] = st.sidebar.selectbox("Pesos", ["uniform", "distance"])
        params["metric"] = st.sidebar.selectbox(
            "Métrica de distancia", ["euclidean", "manhattan", "minkowski"]
        )
    elif clf_key == "tree":
        params["max_depth"] = st.sidebar.slider(
            "Profundidad máxima", 1, 20, 3
        )
        params["criterion"] = st.sidebar.selectbox(
            "Criterio", ["gini", "entropy"]
        )
        params["min_samples_split"] = st.sidebar.slider(
            "Mínimo muestras para dividir", 2, 20, 2
        )
    elif clf_key == "rf":
        params["n_estimators"] = st.sidebar.slider("Número de árboles", 10, 300, 100, step=10)
        params["max_depth"] = st.sidebar.slider(
            "Profundidad máxima", 1, 20, 5
        )
        params["criterion"] = st.sidebar.selectbox(
            "Criterio", ["gini", "entropy"]
        )
        params["random_state"] = int(random_state)
    elif clf_key == "svm":
        params["C"] = st.sidebar.slider("C (regularización)", 0.01, 10.0, 1.0, step=0.01)
        params["kernel"] = st.sidebar.selectbox(
            "Kernel", ["rbf", "linear", "poly", "sigmoid"]
        )
        params["gamma"] = st.sidebar.selectbox("Gamma", ["scale", "auto"])
    elif clf_key == "lr":
        params["C"] = st.sidebar.slider("C (regularización)", 0.01, 10.0, 1.0, step=0.01)
        params["solver"] = st.sidebar.selectbox(
            "Solver", ["lbfgs", "liblinear", "sag", "saga"]
        )
        params["random_state"] = int(random_state)

    return test_size / 100, int(random_state), clf_label, clf_key, params


# ── Sección 1: Introducción ───────────────────────────────────────────────────
def section_intro():
    st.title("🌸 Clasificador del Dataset IRIS")
    st.markdown(
        """
        ## ¿Qué es el dataset IRIS?

        El **dataset IRIS** fue introducido por el estadístico Ronald Fisher en 1936 y es uno de los
        conjuntos de datos más famosos en el aprendizaje automático. Contiene **150 muestras** de
        flores de iris pertenecientes a tres especies distintas:

        | Especie | Características |
        |---|---|
        | 🌼 *Iris setosa* | Pétalos muy pequeños, fácil de distinguir |
        | 🌸 *Iris versicolor* | Tamaño intermedio |
        | 🌺 *Iris virginica* | Pétalos más grandes |

        Cada muestra tiene **4 características** medidas en centímetros:
        - **Largo del sépalo**
        - **Ancho del sépalo**
        - **Largo del pétalo**
        - **Ancho del pétalo**

        El objetivo es **predecir la especie** a partir de estas mediciones.
        """,
        unsafe_allow_html=False,
    )


# ── Sección 2: EDA ────────────────────────────────────────────────────────────
def section_eda(df: pd.DataFrame):
    st.header("📊 Exploración de Datos (EDA)")

    with st.expander("🔍 Vista previa del dataset", expanded=True):
        st.dataframe(df.drop(columns=["target"]), use_container_width=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Total de muestras", len(df))
        c2.metric("Características", 4)
        c3.metric("Clases", 3)

    with st.expander("📈 Estadísticas descriptivas"):
        st.dataframe(df.drop(columns=["target"]).describe(), use_container_width=True)

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Distribución", "Diagrama de caja", "Correlación", "PCA 2D"]
    )

    with tab1:
        feature_sel = st.selectbox(
            "Selecciona una característica", FEATURE_NAMES, key="eda_feat"
        )
        fig = px.histogram(
            df,
            x=feature_sel,
            color="Especie",
            barmode="overlay",
            color_discrete_sequence=PALETTE,
            nbins=25,
            title=f"Distribución de {feature_sel} por especie",
        )
        fig.update_layout(bargap=0.05)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        df_melted = df.melt(
            id_vars="Especie",
            value_vars=FEATURE_NAMES,
            var_name="Característica",
            value_name="Valor (cm)",
        )
        fig = px.box(
            df_melted,
            x="Característica",
            y="Valor (cm)",
            color="Especie",
            color_discrete_sequence=PALETTE,
            title="Diagramas de caja por característica y especie",
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        fig, ax = plt.subplots(figsize=(6, 5))
        corr = df[FEATURE_NAMES].corr()
        sns.heatmap(
            corr,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            ax=ax,
            linewidths=0.5,
        )
        ax.set_title("Mapa de calor de correlaciones")
        st.pyplot(fig)
        plt.close(fig)

    with tab4:
        pca = PCA(n_components=2)
        coords = pca.fit_transform(df[FEATURE_NAMES])
        pca_df = pd.DataFrame(
            coords, columns=["PC1", "PC2"]
        )
        pca_df["Especie"] = df["Especie"]
        var = pca.explained_variance_ratio_ * 100
        fig = px.scatter(
            pca_df,
            x="PC1",
            y="PC2",
            color="Especie",
            color_discrete_sequence=PALETTE,
            title=f"PCA 2D — PC1: {var[0]:.1f}%  PC2: {var[1]:.1f}% de varianza explicada",
            labels={"PC1": f"PC1 ({var[0]:.1f}%)", "PC2": f"PC2 ({var[1]:.1f}%)"},
        )
        st.plotly_chart(fig, use_container_width=True)


# ── Sección 3: Entrenamiento ─────────────────────────────────────────────────
def train_model(df, test_size, random_state, clf_key, params):
    X = df[FEATURE_NAMES].values
    y = df["target"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    clf = build_classifier(clf_key, params)
    clf.fit(X_train, y_train)

    return clf, X_train, X_test, y_train, y_test


# ── Sección 4: Desempeño ─────────────────────────────────────────────────────
def section_performance(clf, clf_label, clf_key, X_train, X_test, y_train, y_test, X_full, y_full):
    st.header("📉 Métricas y Gráficas de Desempeño")

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)

    acc = (y_pred == y_test).mean()
    train_acc = (clf.predict(X_train) == y_train).mean()

    # ── Métricas rápidas ────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    c1.metric("✅ Exactitud (test)", f"{acc:.2%}")
    c2.metric("🏋️ Exactitud (entrenamiento)", f"{train_acc:.2%}")
    c3.metric("🔢 Muestras de prueba", len(y_test))

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Matriz de confusión", "Reporte de clasificación", "Curva ROC", "Curva de aprendizaje", "Importancia / Coeficientes"]
    )

    # ── Tab 1: Matriz de confusión ─────────────────────────────────────────
    with tab1:
        st.subheader("Matriz de Confusión")
        st.markdown(
            "Muestra cuántas muestras de cada clase fueron predichas correctamente (diagonal) "
            "y cuáles fueron confundidas con otras clases."
        )
        fig, ax = plt.subplots(figsize=(5, 4))
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"Matriz de Confusión — {clf_label}")
        st.pyplot(fig)
        plt.close(fig)

    # ── Tab 2: Reporte de clasificación ────────────────────────────────────
    with tab2:
        st.subheader("Reporte de Clasificación")
        st.markdown(
            "**Precisión**: de los predichos como clase X, ¿cuántos realmente lo son?  \n"
            "**Recall**: de todos los reales de clase X, ¿cuántos se detectaron?  \n"
            "**F1-score**: media armónica de precisión y recall."
        )
        report = classification_report(
            y_test, y_pred, target_names=CLASS_NAMES, output_dict=True
        )
        report_df = pd.DataFrame(report).T.round(3)
        st.dataframe(report_df, use_container_width=True)

    # ── Tab 3: Curva ROC ────────────────────────────────────────────────────
    with tab3:
        st.subheader("Curva ROC (One-vs-Rest)")
        st.markdown(
            "La curva ROC muestra la relación entre la tasa de verdaderos positivos y la de "
            "falsos positivos para cada clase. Un AUC cercano a 1 indica mejor desempeño."
        )
        y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
        fig = go.Figure()
        for i, cls in enumerate(CLASS_NAMES):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            fig.add_trace(
                go.Scatter(
                    x=fpr,
                    y=tpr,
                    mode="lines",
                    name=f"{cls} (AUC={roc_auc:.3f})",
                    line=dict(color=PALETTE[i], width=2),
                )
            )
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                line=dict(dash="dash", color="gray"),
                showlegend=False,
            )
        )
        fig.update_layout(
            xaxis_title="Tasa de Falsos Positivos",
            yaxis_title="Tasa de Verdaderos Positivos",
            title="Curva ROC por clase",
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1.05]),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Tab 4: Curva de aprendizaje ─────────────────────────────────────────
    with tab4:
        st.subheader("Curva de Aprendizaje")
        st.markdown(
            "Muestra cómo evoluciona la exactitud del modelo al aumentar el tamaño del "
            "conjunto de entrenamiento. Permite detectar **sobreajuste** (train >> val) "
            "o **subajuste** (ambas curvas bajas)."
        )
        train_sizes, train_scores, val_scores = learning_curve(
            clf,
            X_full,
            y_full,
            cv=5,
            n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring="accuracy",
        )
        train_mean = train_scores.mean(axis=1)
        train_std = train_scores.std(axis=1)
        val_mean = val_scores.mean(axis=1)
        val_std = val_scores.std(axis=1)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=train_sizes,
                y=train_mean,
                mode="lines+markers",
                name="Entrenamiento",
                line=dict(color="#4C72B0", width=2),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([train_sizes, train_sizes[::-1]]),
                y=np.concatenate([train_mean + train_std, (train_mean - train_std)[::-1]]),
                fill="toself",
                fillcolor="rgba(76,114,176,0.15)",
                line=dict(color="rgba(255,255,255,0)"),
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=train_sizes,
                y=val_mean,
                mode="lines+markers",
                name="Validación cruzada",
                line=dict(color="#DD8452", width=2),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([train_sizes, train_sizes[::-1]]),
                y=np.concatenate([val_mean + val_std, (val_mean - val_std)[::-1]]),
                fill="toself",
                fillcolor="rgba(221,132,82,0.15)",
                line=dict(color="rgba(255,255,255,0)"),
                showlegend=False,
            )
        )
        fig.update_layout(
            xaxis_title="Tamaño del conjunto de entrenamiento",
            yaxis_title="Exactitud",
            title="Curva de Aprendizaje",
            yaxis=dict(range=[0, 1.05]),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Tab 5: Importancia / coeficientes ───────────────────────────────────
    with tab5:
        st.subheader("Importancia de Características / Coeficientes")

        if clf_key in ("tree", "rf"):
            importances = clf.feature_importances_
            fig = px.bar(
                x=importances,
                y=FEATURE_NAMES,
                orientation="h",
                labels={"x": "Importancia", "y": "Característica"},
                title="Importancia de características (impureza Gini)",
                color=importances,
                color_continuous_scale="Blues",
            )
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

            if clf_key == "tree":
                st.markdown("**Representación textual del árbol:**")
                tree_text = export_text(clf, feature_names=FEATURE_NAMES)
                st.code(tree_text, language="text")

                st.markdown("**Visualización del árbol de decisión:**")
                fig2, ax2 = plt.subplots(figsize=(12, 6))
                plot_tree(
                    clf,
                    feature_names=FEATURE_NAMES,
                    class_names=CLASS_NAMES,
                    filled=True,
                    rounded=True,
                    ax=ax2,
                    fontsize=9,
                )
                st.pyplot(fig2)
                plt.close(fig2)

        elif clf_key == "lr":
            coef_df = pd.DataFrame(
                clf.coef_,
                columns=FEATURE_NAMES,
                index=CLASS_NAMES,
            )
            fig = px.imshow(
                coef_df,
                text_auto=".2f",
                aspect="auto",
                color_continuous_scale="RdBu",
                title="Coeficientes de Regresión Logística",
            )
            st.plotly_chart(fig, use_container_width=True)

        elif clf_key == "svm" and clf.kernel == "linear":
            coef = np.abs(clf.coef_).mean(axis=0)
            fig = px.bar(
                x=coef,
                y=FEATURE_NAMES,
                orientation="h",
                labels={"x": "Peso promedio |w|", "y": "Característica"},
                title="Importancia de características (SVM lineal)",
                color=coef,
                color_continuous_scale="Blues",
            )
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.info(
                "La importancia de características directa no está disponible para este "
                "clasificador. Intenta con Árbol de Decisión, Random Forest o Regresión Logística."
            )


# ── Sección 5: Predicción interactiva ────────────────────────────────────────
def section_prediction(clf):
    st.header("🔮 Predicción Interactiva")
    st.markdown(
        "Introduce los valores de las características de una flor y el modelo clasificará "
        "a qué especie pertenece."
    )

    iris_raw = load_iris()
    X_all = iris_raw.data

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Ingresa las características")
        inputs = {}
        for fname, fkey, col_idx in zip(FEATURE_NAMES, FEATURE_KEYS, range(4)):
            min_v = float(X_all[:, col_idx].min())
            max_v = float(X_all[:, col_idx].max())
            default = float(X_all[:, col_idx].mean())
            inputs[fkey] = st.slider(
                fname,
                min_value=round(min_v - 0.5, 1),
                max_value=round(max_v + 0.5, 1),
                value=round(default, 1),
                step=0.1,
                key=f"pred_{fkey}",
            )

        sample = np.array(
            [[inputs[k] for k in FEATURE_KEYS]]
        )

        predict_btn = st.button("🔍 Predecir especie", use_container_width=True)

    with col2:
        st.subheader("Resultado de la predicción")
        if predict_btn:
            prediction = clf.predict(sample)[0]
            proba = clf.predict_proba(sample)[0]
            especie = CLASS_NAMES[prediction]
            emojis = ["🌼", "🌸", "🌺"]

            st.success(f"### {emojis[prediction]} Especie predicha: **{especie}**")

            prob_df = pd.DataFrame(
                {"Especie": CLASS_NAMES, "Probabilidad": proba}
            )
            fig = px.bar(
                prob_df,
                x="Especie",
                y="Probabilidad",
                color="Especie",
                color_discrete_sequence=PALETTE,
                title="Probabilidades por clase",
                range_y=[0, 1],
                text=prob_df["Probabilidad"].map(lambda p: f"{p:.1%}"),
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("**Valores ingresados:**")
            input_df = pd.DataFrame(
                [list(inputs.values())], columns=FEATURE_NAMES
            )
            st.dataframe(input_df, use_container_width=True)
        else:
            st.info("Ajusta los controles deslizantes y presiona **Predecir especie**.")


# ── Punto de entrada ──────────────────────────────────────────────────────────
def main():
    df, _iris = load_data()
    test_size, random_state, clf_label, clf_key, params = sidebar()

    section_intro()

    st.divider()
    section_eda(df)

    st.divider()
    with st.spinner("Entrenando el modelo…"):
        clf, X_train, X_test, y_train, y_test = train_model(
            df, test_size, random_state, clf_key, params
        )

    st.header(f"🤖 Clasificador: {clf_label}")
    st.markdown(
        f"Modelo entrenado con **{len(X_train)}** muestras y evaluado con **{len(X_test)}** muestras."
    )

    st.divider()
    section_performance(clf, clf_label, clf_key, X_train, X_test, y_train, y_test,
                        df[FEATURE_NAMES].values, df["target"].values)

    st.divider()
    section_prediction(clf)


if __name__ == "__main__":
    main()
