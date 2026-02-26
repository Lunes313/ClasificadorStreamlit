import warnings
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st
from PIL import Image
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from streamlit_drawable_canvas import st_canvas

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Clasificador MNIST Interactivo",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@dataclass
class ModelResult:
    name: str
    estimator: object
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: np.ndarray


@st.cache_data(show_spinner=True)
def load_mnist_data() -> tuple[np.ndarray, np.ndarray]:
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    X = mnist.data.astype(np.float32) / 255.0
    y = mnist.target.astype(np.int64)
    return X, y


def build_models(random_state: int) -> dict[str, object]:
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=350, random_state=random_state, n_jobs=None),
        ),
        "SVM (RBF)": make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", C=3.0, gamma="scale", probability=True, random_state=random_state),
        ),
        "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
        "Random Forest": RandomForestClassifier(
            n_estimators=160,
            max_depth=20,
            min_samples_split=2,
            random_state=random_state,
            n_jobs=-1,
        ),
        "Red Neuronal Simple (MLP)": make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(128,),
                activation="relu",
                solver="adam",
                learning_rate_init=0.001,
                batch_size=256,
                max_iter=25,
                random_state=random_state,
            ),
        ),
    }


@st.cache_resource(show_spinner=True)
def train_models(sample_size: int, test_size: float, random_state: int):
    X, y = load_mnist_data()

    if sample_size < len(X):
        X_subset, _, y_subset, _ = train_test_split(
            X,
            y,
            train_size=sample_size,
            random_state=random_state,
            stratify=y,
        )
    else:
        X_subset, y_subset = X, y

    X_train, X_valid, y_train, y_valid = train_test_split(
        X_subset,
        y_subset,
        test_size=test_size,
        random_state=random_state,
        stratify=y_subset,
    )

    model_results: list[ModelResult] = []

    for model_name, model in build_models(random_state).items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_valid)

        model_results.append(
            ModelResult(
                name=model_name,
                estimator=model,
                accuracy=accuracy_score(y_valid, y_pred),
                precision=precision_score(y_valid, y_pred, average="macro", zero_division=0),
                recall=recall_score(y_valid, y_pred, average="macro", zero_division=0),
                f1=f1_score(y_valid, y_pred, average="macro", zero_division=0),
                confusion_matrix=confusion_matrix(y_valid, y_pred, labels=np.arange(10)),
            )
        )

    metrics_df = pd.DataFrame(
        {
            "Modelo": [m.name for m in model_results],
            "Accuracy": [m.accuracy for m in model_results],
            "Precision": [m.precision for m in model_results],
            "Recall": [m.recall for m in model_results],
            "F1-score": [m.f1 for m in model_results],
        }
    )

    return model_results, metrics_df, X_valid, y_valid


def preprocess_canvas_image(image_data: np.ndarray) -> np.ndarray:
    rgba_image = Image.fromarray(image_data.astype("uint8"), mode="RGBA")
    gray_image = rgba_image.convert("L")
    resized = gray_image.resize((28, 28), Image.Resampling.LANCZOS)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    return arr.reshape(1, -1), arr


def render_confusion_matrix(matrix: np.ndarray, title: str):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=False,
        cmap="Blues",
        cbar=True,
        ax=ax,
        linewidths=0.2,
        linecolor="white",
    )
    ax.set_title(title)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Etiqueta real")
    ax.set_xticks(np.arange(10) + 0.5)
    ax.set_yticks(np.arange(10) + 0.5)
    ax.set_xticklabels(np.arange(10))
    ax.set_yticklabels(np.arange(10))
    st.pyplot(fig)
    plt.close(fig)


def main():
    st.title("✍️ Clasificador de Dígitos MNIST — Interactivo y Pedagógico")
    st.caption(
        "Modelos incluidos: Logistic Regression, SVM, KNN, Random Forest y Red Neuronal Simple (MLP)."
    )

    st.sidebar.header("⚙️ Configuración")
    sample_size = st.sidebar.slider("Tamaño de muestra para entrenamiento", 5000, 70000, 12000, step=1000)
    test_size_pct = st.sidebar.slider("Porcentaje de validación", 10, 40, 20, step=5)
    random_state = st.sidebar.number_input("Semilla aleatoria", min_value=0, max_value=9999, value=42, step=1)

    with st.spinner("Entrenando modelos y evaluando desempeño..."):
        model_results, metrics_df, X_valid, y_valid = train_models(
            sample_size=sample_size,
            test_size=test_size_pct / 100,
            random_state=int(random_state),
        )

    st.success("Entrenamiento completado. Ya puedes comparar modelos y hacer predicciones.")

    tab1, tab2, tab3 = st.tabs(
        [
            "📊 Desempeño de Modelos",
            "🧪 Probar con imagen de validación",
            "🎨 Dibujar dígito en tiempo real",
        ]
    )

    with tab1:
        st.subheader("Métricas de desempeño")
        st.dataframe(
            metrics_df.style.format(
                {
                    "Accuracy": "{:.4f}",
                    "Precision": "{:.4f}",
                    "Recall": "{:.4f}",
                    "F1-score": "{:.4f}",
                }
            ),
            use_container_width=True,
        )

        metrics_long = metrics_df.melt(id_vars="Modelo", var_name="Métrica", value_name="Valor")
        fig = px.bar(
            metrics_long,
            x="Modelo",
            y="Valor",
            color="Métrica",
            barmode="group",
            title="Comparativa de métricas por modelo",
            text_auto=".3f",
        )
        fig.update_layout(yaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)

        selected_model_name = st.selectbox(
            "Selecciona un modelo para ver su matriz de confusión",
            [m.name for m in model_results],
            key="model_cm_select",
        )
        selected_result = next(m for m in model_results if m.name == selected_model_name)
        render_confusion_matrix(
            selected_result.confusion_matrix,
            f"Matriz de confusión — {selected_result.name}",
        )

    with tab2:
        st.subheader("Predicción sobre imágenes reales del conjunto de validación")
        selected_model_name = st.selectbox(
            "Modelo para predecir",
            [m.name for m in model_results],
            key="model_pred_select",
        )
        selected_result = next(m for m in model_results if m.name == selected_model_name)

        img_index = st.slider("Selecciona índice de imagen", 0, len(X_valid) - 1, 0)
        image_flat = X_valid[img_index]
        true_label = int(y_valid[img_index])
        pred_label = int(selected_result.estimator.predict(image_flat.reshape(1, -1))[0])

        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(image_flat.reshape(28, 28), caption="Imagen seleccionada (MNIST)", width=280)
        with col2:
            st.metric("Etiqueta real", true_label)
            st.metric("Predicción del modelo", pred_label)
            st.write(
                "Correcto ✅" if true_label == pred_label else "Incorrecto ❌"
            )

    with tab3:
        st.subheader("Dibuja un dígito y clasifícalo")
        st.write(
            "Dibuja en blanco sobre fondo negro. La predicción se actualiza automáticamente en cada cambio."
        )

        selected_model_name = st.selectbox(
            "Modelo para predecir dibujo",
            [m.name for m in model_results],
            key="model_draw_select",
        )
        selected_result = next(m for m in model_results if m.name == selected_model_name)

        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 1)",
            stroke_width=18,
            stroke_color="#FFFFFF",
            background_color="#000000",
            width=280,
            height=280,
            drawing_mode="freedraw",
            key="canvas",
            update_streamlit=True,
        )

        if canvas_result.image_data is not None:
            image_data = canvas_result.image_data
            drawn_flat, drawn_img = preprocess_canvas_image(image_data)
            pred_digit = int(selected_result.estimator.predict(drawn_flat)[0])

            c1, c2 = st.columns([1, 1])
            with c1:
                st.image(drawn_img, caption="Tu dibujo normalizado (28x28)", width=180, clamp=True)
            with c2:
                st.metric("Predicción del dígito", pred_digit)

                if hasattr(selected_result.estimator, "predict_proba"):
                    probs = selected_result.estimator.predict_proba(drawn_flat)[0]
                    prob_df = pd.DataFrame(
                        {"Dígito": np.arange(10), "Probabilidad": probs}
                    )
                    fig_prob = px.bar(
                        prob_df,
                        x="Dígito",
                        y="Probabilidad",
                        title="Probabilidad por clase",
                        range_y=[0, 1],
                    )
                    st.plotly_chart(fig_prob, use_container_width=True)


if __name__ == "__main__":
    main()
