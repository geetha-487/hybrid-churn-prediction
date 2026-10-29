
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import pennylane as qml
import shap

# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Hybrid Quantum Churn AI",
    page_icon="⚛️",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = "churn_model_artifacts"

st.markdown("""
<style>
.main-title {
    font-size: 2.2rem;
    font-weight: 750;
}
.subtitle {
    color: #8b95a5;
    font-size: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# 2. LOAD SAVED MODELS AND CONFIGURATION
# ============================================================
@st.cache_resource
def load_artifacts():
    model = joblib.load(ARTIFACTS / "classical_baseline.joblib")
    scaler = joblib.load(ARTIFACTS / "quantum_feature_scaler.joblib")

    with open(ARTIFACTS / "feature_names.json", encoding="utf-8") as f:
        feature_names = json.load(f)

    with open(ARTIFACTS / "quantum_config.json", encoding="utf-8") as f:
        config = json.load(f)

    with open(ARTIFACTS / "evaluation_results.json", encoding="utf-8") as f:
        evaluation = json.load(f)

    weights = np.load(ARTIFACTS / "quantum_weights.npy")

    return model, scaler, feature_names, config, evaluation, weights


try:
    model, scaler, feature_names, config, evaluation, weights = load_artifacts()
except Exception as exc:
    st.error(f"Could not load saved model files: {exc}")
    st.info("Check that all model artifacts are inside the model_artifacts folder.")
    st.stop()

n_qubits = int(config["n_qubits"])
threshold = float(config.get("probability_threshold", 0.5))

if len(weights) != n_qubits or len(feature_names) != n_qubits:
    st.error("The saved feature count, qubit count, and weights do not match.")
    st.stop()

# ============================================================
# 3. QUANTUM CIRCUIT
# ============================================================
device = qml.device("default.qubit", wires=n_qubits)


@qml.qnode(device, interface="numpy", diff_method=None)
def quantum_circuit(inputs, trainable_weights):
    # RY feature encoding
    for i in range(n_qubits):
        qml.RY(inputs[i], wires=i)

    # Trainable RY and RZ rotations
    for i in range(n_qubits):
        qml.RY(trainable_weights[i], wires=i)
        qml.RZ(trainable_weights[i], wires=i)

    # Nearest-neighbour CNOT entanglement
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i + 1])

    # Pauli-Z measurement
    return qml.expval(qml.PauliZ(0))


def quantum_probability(scaled_features):
    expectation = float(
        quantum_circuit(
            np.asarray(scaled_features, dtype=float),
            np.asarray(weights, dtype=float)
        )
    )

    # Assumes the original notebook used this probability mapping.
    probability = (expectation + 1.0) / 2.0

    return float(np.clip(probability, 0.0, 1.0)), expectation


def classical_probability(scaled_features):
    return float(
        model.predict_proba(
            np.asarray(scaled_features).reshape(1, -1)
        )[0, 1]
    )


# ============================================================
# 4. REFERENCE TEST METRICS
# ============================================================
metrics = {
    "Quantum": {
        "Accuracy": 0.6544,
        "Precision": 0.4158,
        "Recall": 0.7460,
        "F1-score": 0.5340,
        "ROC-AUC": 0.7509
    },
    "Classical": {
        "Accuracy": 0.7445,
        "Precision": 0.5127,
        "Recall": 0.7567,
        "F1-score": 0.6112,
        "ROC-AUC": 0.8243
    }
}

# ============================================================
# 5. HEADER AND NAVIGATION
# ============================================================
st.markdown(
    '<div class="main-title">⚛️ Hybrid Quantum–Classical Churn AI</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="subtitle">Telecom Customer Churn Prediction | '
    'Quantum ML | Explainable AI</div>',
    unsafe_allow_html=True
)

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Choose a section",
    [
        "Dashboard",
        "Customer Prediction",
        "Model Comparison",
        "Explainable AI",
        "Project Details"
    ]
)

st.sidebar.divider()
st.sidebar.caption(f"Quantum circuit: {n_qubits} qubits")
st.sidebar.caption(f"Training epochs: {config.get('epochs', 5)}")
st.sidebar.caption(
    f"Training sample: {config.get('training_sample_size', 'N/A')}"
)

# ============================================================
# 6. DASHBOARD
# ============================================================
if page == "Dashboard":
    st.subheader("Model Performance Dashboard")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Quantum Qubits", n_qubits)
    c2.metric("Training Epochs", config.get("epochs", 5))
    c3.metric("Quantum ROC-AUC", f"{metrics['Quantum']['ROC-AUC']:.2%}")
    c4.metric("Classical ROC-AUC", f"{metrics['Classical']['ROC-AUC']:.2%}")

    st.divider()

    left, right = st.columns(2)

    with left:
        st.markdown("#### Accuracy comparison")
        df = pd.DataFrame({
            "Model": ["Quantum", "Classical"],
            "Accuracy": [
                metrics["Quantum"]["Accuracy"],
                metrics["Classical"]["Accuracy"]
            ]
        }).set_index("Model")
        st.bar_chart(df)

    with right:
        st.markdown("#### ROC-AUC comparison")
        df = pd.DataFrame({
            "Model": ["Quantum", "Classical"],
            "ROC-AUC": [
                metrics["Quantum"]["ROC-AUC"],
                metrics["Classical"]["ROC-AUC"]
            ]
        }).set_index("Model")
        st.bar_chart(df)

    st.markdown("#### Evaluation metrics")
    st.dataframe(
        pd.DataFrame(metrics).T.style.format("{:.2%}"),
        use_container_width=True
    )

    st.info(
        "These are recorded test-set results from the notebook, not newly "
        "calculated metrics. The quantum model is a preliminary prototype."
    )

# ============================================================
# 7. CUSTOMER PREDICTION
# ============================================================
elif page == "Customer Prediction":
    st.subheader("Predict Customer Churn")

    with st.form("customer_form"):
        tenure = st.slider(
            "Customer tenure (months)", 0, 72, 24
        )

        internet = st.selectbox(
            "Internet service",
            ["DSL", "Fiber optic", "No"]
        )

        payment = st.selectbox(
            "Payment method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer",
                "Credit card"
            ]
        )

        contract = st.selectbox(
            "Contract type",
            ["Month-to-month", "One year", "Two year"]
        )

        submitted = st.form_submit_button(
            "Run Hybrid Prediction",
            use_container_width=True
        )

    if submitted:
        feature_values = {
            "tenure": float(tenure),
            "InternetService_Fiber optic": float(internet == "Fiber optic"),
            "PaymentMethod_Electronic check": float(
                payment == "Electronic check"
            ),
            "Contract_Two year": float(contract == "Two year")
        }

        row = np.array(
            [[feature_values[name] for name in feature_names]],
            dtype=float
        )

        try:
            scaled = scaler.transform(row)[0]
            q_prob, q_expectation = quantum_probability(scaled)
            c_prob = classical_probability(scaled)

            result = {
                "features": feature_values,
                "scaled": scaled,
                "quantum_probability": q_prob,
                "quantum_expectation": q_expectation,
                "quantum_class": int(q_prob >= threshold),
                "classical_probability": c_prob,
                "classical_class": int(c_prob >= threshold)
            }

            st.session_state["last_prediction"] = result

            if "prediction_history" not in st.session_state:
                st.session_state["prediction_history"] = []

            st.session_state["prediction_history"].append({
                "Tenure": tenure,
                "Internet": internet,
                "Payment": payment,
                "Contract": contract,
                "Quantum score": q_prob,
                "Classical score": c_prob
            })

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

    result = st.session_state.get("last_prediction")

    if result:
        st.divider()
        left, right = st.columns(2)

        with left:
            st.markdown("### ⚛️ Quantum prediction")
            st.metric("Quantum score", f"{result['quantum_probability']:.1%}")
            st.progress(result["quantum_probability"])
            st.write(
                "Likely to churn"
                if result["quantum_class"]
                else "Unlikely to churn"
            )
            st.caption(
                f"Pauli-Z expectation: {result['quantum_expectation']:.4f}"
            )

        with right:
            st.markdown("### Classical prediction")
            st.metric(
                "Churn probability",
                f"{result['classical_probability']:.1%}"
            )
            st.progress(result["classical_probability"])
            st.write(
                "Likely to churn"
                if result["classical_class"]
                else "Unlikely to churn"
            )

        st.markdown("#### Customer risk guidance")
        highest_score = max(
            result["quantum_probability"],
            result["classical_probability"]
        )

        if highest_score >= 0.70:
            st.warning(
                "At least one model indicates elevated risk. Review customer "
                "feedback and service concerns before deciding on retention action."
            )
        elif highest_score >= 0.40:
            st.info(
                "The scores suggest possible risk. Monitor the customer and "
                "investigate any service concerns."
            )
        else:
            st.success(
                "Both scores are below the illustrative elevated-risk range."
            )

        st.caption(
            "These are model scores, not guarantees. Risk bands are illustrative."
        )

        st.markdown("#### Model score comparison")
        comparison = pd.DataFrame({
            "Model": ["Quantum", "Classical"],
            "Churn score": [
                result["quantum_probability"],
                result["classical_probability"]
            ]
        }).set_index("Model")
        st.bar_chart(comparison)

    if st.session_state.get("prediction_history"):
        st.markdown("#### Prediction history")
        history = pd.DataFrame(st.session_state["prediction_history"])
        st.dataframe(history, use_container_width=True, hide_index=True)

        st.download_button(
            "Download prediction history CSV",
            data=history.to_csv(index=False).encode("utf-8"),
            file_name="churn_prediction_history.csv",
            mime="text/csv"
        )

# ============================================================
# 8. MODEL COMPARISON
# ============================================================
elif page == "Model Comparison":
    st.subheader("Quantum vs Classical Model Evaluation")

    selected_metric = st.selectbox(
        "Choose evaluation metric",
        ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]
    )

    comparison_df = pd.DataFrame({
        "Model": ["Quantum", "Classical"],
        selected_metric: [
            metrics["Quantum"][selected_metric],
            metrics["Classical"][selected_metric]
        ]
    }).set_index("Model")

    st.bar_chart(comparison_df)

    st.dataframe(
        pd.DataFrame(metrics).T.style.format("{:.2%}"),
        use_container_width=True
    )

    st.warning(
        "The recorded classical model performs better on all five listed "
        "metrics. These results do not establish quantum advantage."
    )

# ============================================================
# 9. EXPLAINABLE AI WITH SHAP
# ============================================================
elif page == "Explainable AI":
    st.subheader("Explainable AI — SHAP Analysis")

    st.write(
        "SHAP explains how input features influence the classical "
        "Logistic Regression model's output."
    )

    result = st.session_state.get("last_prediction")

    if result is None:
        st.info(
            "Go to Customer Prediction, run a prediction, and return here "
            "to view its SHAP explanation."
        )
    else:
        try:
            customer = np.asarray(
                result["scaled"], dtype=float
            ).reshape(1, -1)

            # LinearExplainer explains the fitted linear model.
            # Background is zero in scaled space, representing the training
            # mean for a StandardScaler.
            background = np.zeros((1, len(feature_names)))

            explainer = shap.LinearExplainer(model, background)
            explanation = explainer(customer)

            values = np.asarray(explanation.values)

            if values.ndim == 3:
                # Select the positive class if SHAP returns class outputs.
                values = values[0, :, 1]
            elif values.ndim == 2:
                values = values[0]
            else:
                values = values.reshape(-1)

            if len(values) != len(feature_names):
                st.error(
                    f"Unexpected SHAP output shape: "
                    f"{np.asarray(explanation.values).shape}"
                )
                st.stop()

            shap_df = pd.DataFrame({
                "Feature": feature_names,
                "SHAP value": values
            })
            shap_df["Absolute impact"] = shap_df["SHAP value"].abs()
            shap_df = shap_df.sort_values(
                "Absolute impact", ascending=False
            )

            st.markdown("### 1. Feature contribution chart")
            st.caption(
                "Positive SHAP values push the model output toward churn; "
                "negative values push it away from churn. For this linear "
                "model, values are normally on the log-odds scale."
            )

            st.bar_chart(
                shap_df.set_index("Feature")[["SHAP value"]]
            )

            st.markdown("### 2. Feature contribution table")
            st.dataframe(
                shap_df[["Feature", "SHAP value"]],
                use_container_width=True,
                hide_index=True
            )

            st.markdown("### 3. Customer prediction summary")

            base_values = np.asarray(
                explanation.base_values
            ).reshape(-1)
            base_value = float(base_values[0])

            c1, c2 = st.columns(2)
            c1.metric("SHAP baseline output", f"{base_value:.4f}")
            c2.metric(
                "Classical churn probability",
                f"{result['classical_probability']:.1%}"
            )

            top_feature = shap_df.iloc[0]

            if top_feature["SHAP value"] > 0:
                direction = "toward a higher model output"
            elif top_feature["SHAP value"] < 0:
                direction = "toward a lower model output"
            else:
                direction = "with zero contribution"

            st.markdown("### 4. Interpretation")
            st.write(
                f"The largest absolute SHAP contribution for this customer "
                f"is **{top_feature['Feature']}**, pushing the model output "
                f"{direction}."
            )

            st.caption(
                "SHAP values and the baseline are not churn percentages. "
                "They explain the model output and should not be added "
                "directly to the displayed probability."
            )

        except Exception as exc:
            st.error(f"SHAP explanation failed: {exc}")
            st.info(
                "Copy the complete error message and send it to me "
                "so I can help correct it."
            )

    st.divider()
    st.markdown("### Quantum model sensitivity analysis")

    st.write(
        "This separate analysis changes one scaled feature at a time from "
        "a zero-vector reference. It is a sensitivity analysis, not SHAP."
    )

    if result is not None:
        reference = np.zeros(n_qubits, dtype=float)
        reference_probability, _ = quantum_probability(reference)
        scaled = np.asarray(result["scaled"], dtype=float)
        rows = []

        for i, name in enumerate(feature_names):
            changed = reference.copy()
            changed[i] = scaled[i]
            changed_probability, _ = quantum_probability(changed)

            rows.append({
                "Feature": name,
                "Reference score": reference_probability,
                "One-feature score": changed_probability,
                "Score change": changed_probability - reference_probability
            })

        sensitivity_df = pd.DataFrame(rows)

        st.bar_chart(
            sensitivity_df.set_index("Feature")[["Score change"]]
        )
        st.dataframe(
            sensitivity_df,
            use_container_width=True,
            hide_index=True
        )

# ============================================================
# 10. PROJECT DETAILS
# ============================================================
elif page == "Project Details":
    st.subheader("Project Architecture")

    st.markdown("""
    **Telecom customer input**

    ↓

    **Feature selection and scaling**

    ↓

    **Parallel model predictions**
    - Classical Logistic Regression
    - Four-qubit PennyLane quantum circuit

    ↓

    **Prediction comparison and dashboard**

    ↓

    **Explainable AI**
    - SHAP for the classical model
    - One-feature sensitivity analysis for the quantum model
    """)

    st.markdown("#### Quantum configuration")
    st.json(config)

    st.markdown("#### Saved quantum weights")
    st.write(weights.tolist())

    st.markdown("#### Research limitations")
    st.write(
        "- The quantum model was trained on 160 samples for five epochs."
    )
    st.write(
        "- The classical baseline achieved better recorded test metrics."
    )
    st.write(
        "- The quantum circuit runs on a local simulator, not physical quantum hardware."
    )
    st.write(
        "- The quantum sensitivity analysis is not a validated SHAP explanation."
    )

st.divider()
st.caption(
    "Hybrid Quantum–Classical Telecom Churn Prediction | Academic Prototype"
)
