
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

try:
    import pennylane as qml
except Exception:
    qml = None

try:
    import shap
except Exception:
    shap = None


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Hybrid Churn Prediction",
    page_icon="📡",
    layout="wide"
)

st.title("📡 Hybrid Quantum–Classical Customer Churn Prediction")
st.caption(
    "Telecom customer churn prediction with explainable AI"
)


# --------------------------------------------------
# FIND SAVED MODEL FILES
# Supports both possible artifact folder names.
# Also searches inside subfolders.
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

ARTIFACT_DIRS = [
    BASE_DIR / "churn_model_artifacts",
    BASE_DIR / "model_artifacts",
]


def find_artifact(filename):
    for folder in ARTIFACT_DIRS:
        if folder.exists():
            candidate = folder / filename
            if candidate.is_file():
                return candidate

            matches = list(folder.rglob(filename))
            if matches:
                return matches[0]

    # Final fallback: search the project directory.
    matches = [
        p for p in BASE_DIR.rglob(filename)
        if ".git" not in p.parts
    ]
    if matches:
        return matches[0]

    raise FileNotFoundError(
        f"Cannot find {filename}. "
        "Check that it is uploaded to your GitHub repository."
    )


# --------------------------------------------------
# LOAD SAVED ARTIFACTS
# --------------------------------------------------
@st.cache_resource
def load_artifacts():
    model_path = find_artifact("classical_baseline.joblib")
    scaler_path = find_artifact("quantum_feature_scaler.joblib")
    weights_path = find_artifact("quantum_weights.npy")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    weights = np.load(weights_path)

    try:
        feature_path = find_artifact("feature_names.json")
        with open(feature_path, "r", encoding="utf-8") as file:
            features = json.load(file)
    except FileNotFoundError:
        features = [
            "tenure",
            "InternetService_Fiber optic",
            "PaymentMethod_Electronic check",
            "Contract_Two year",
        ]

    if isinstance(features, dict):
        features = features.get("feature_names", list(features.keys()))

    try:
        config_path = find_artifact("quantum_config.json")
        with open(config_path, "r", encoding="utf-8") as file:
            config = json.load(file)
    except FileNotFoundError:
        config = {"n_qubits": 4}

    return model, scaler, weights, features, config


try:
    classical_model, scaler, quantum_weights, feature_names, config = (
        load_artifacts()
    )
    st.success("Saved model files loaded successfully.")
except Exception as error:
    st.error(f"Could not load saved model files: {error}")
    st.info(
        "Check that classical_baseline.joblib, "
        "quantum_feature_scaler.joblib, quantum_weights.npy, "
        "and the JSON files are uploaded to your GitHub repository."
    )
    st.stop()


# --------------------------------------------------
# QUANTUM CIRCUIT
# This follows the saved 4-qubit circuit configuration.
# --------------------------------------------------
def quantum_score(scaled_features):
    if qml is None:
        raise RuntimeError(
            "PennyLane is unavailable. Check requirements.txt "
            "and the Streamlit deployment logs."
        )

    n_qubits = int(config.get("n_qubits", 4))

    if n_qubits != 4 or len(scaled_features) != 4:
        raise ValueError(
            "This app expects the trained four-feature, four-qubit model."
        )

    device = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(device)
    def circuit(inputs, weights):
        for i in range(n_qubits):
            qml.RY(inputs[i], wires=i)

        for i in range(n_qubits):
            qml.RY(weights[i], wires=i)
            qml.RZ(weights[i], wires=i)

        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])

        return qml.expval(qml.PauliZ(0))

    expectation = float(
        circuit(
            np.asarray(scaled_features, dtype=float),
            np.asarray(quantum_weights, dtype=float),
        )
    )

    # Converts the Pauli-Z expectation into a 0-to-1 score.
    # This score is not guaranteed to be a calibrated probability.
    return float(np.clip((expectation + 1.0) / 2.0, 0.0, 1.0))


# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------
page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Customer Prediction",
        "Model Comparison",
        "Explainable AI",
        "Project Details",
    ],
)

st.sidebar.divider()
st.sidebar.caption("Hybrid Quantum–Classical ML")
st.sidebar.caption("PennyLane simulator + classical ML + SHAP")


# --------------------------------------------------
# PRELIMINARY NOTEBOOK EVALUATION RESULTS
# These are the previously recorded results.
# They are not recalculated by this website.
# --------------------------------------------------
quantum_metrics = {
    "Accuracy": 0.6544,
    "Precision": 0.4158,
    "Recall": 0.7460,
    "F1-score": 0.5340,
    "ROC-AUC": 0.7509,
}

classical_metrics = {
    "Accuracy": 0.7445,
    "Precision": 0.5127,
    "Recall": 0.7567,
    "F1-score": 0.6112,
    "ROC-AUC": 0.8243,
}


# --------------------------------------------------
# DASHBOARD
# --------------------------------------------------
if page == "Dashboard":
    st.header("Project Dashboard")

    st.write(
        "This project compares a four-qubit quantum circuit "
        "with a classical Logistic Regression baseline for "
        "telecom customer churn prediction."
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Dataset records", "7,043")
    col2.metric("Selected features", "4")
    col3.metric("Quantum circuit", "4 qubits")

    st.subheader("Previously recorded evaluation results")

    comparison = pd.DataFrame(
        {
            "Metric": list(quantum_metrics.keys()),
            "Quantum model": list(quantum_metrics.values()),
            "Classical baseline": list(classical_metrics.values()),
        }
    ).set_index("Metric")

    st.dataframe(
        comparison.style.format("{:.2%}"),
        use_container_width=True,
    )

    st.bar_chart(comparison)

    st.warning(
        "These are the preliminary results recorded from the notebook. "
        "The classical baseline performed better on accuracy, F1-score, "
        "and ROC-AUC. Do not interpret this as demonstrated quantum "
        "advantage."
    )


# --------------------------------------------------
# CUSTOMER PREDICTION
# --------------------------------------------------
elif page == "Customer Prediction":
    st.header("Predict Customer Churn")

    st.write(
        "Enter a customer's details to compare the saved classical model "
        "with the quantum circuit simulator."
    )

    with st.form("customer_form"):
        tenure = st.slider(
            "Customer tenure (months)",
            min_value=0,
            max_value=72,
            value=12,
        )

        fiber = st.selectbox(
            "Internet service",
            ["DSL", "Fiber optic", "No internet service"],
        )

        payment = st.selectbox(
            "Payment method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )

        contract = st.selectbox(
            "Contract type",
            ["Month-to-month", "One year", "Two year"],
        )

        submitted = st.form_submit_button(
            "Predict Churn",
            use_container_width=True,
        )

    if submitted:
        # Match the exact saved feature order.
        raw = np.array(
            [[
                tenure,
                int(fiber == "Fiber optic"),
                int(payment == "Electronic check"),
                int(contract == "Two year"),
            ]],
            dtype=float,
        )

        if len(feature_names) != raw.shape[1]:
            st.error(
                "The saved feature_names.json does not contain "
                "the expected four features."
            )
            st.stop()

        try:
            scaled = scaler.transform(raw)

            # Classical prediction from the saved model.
            classical_probability = float(
                classical_model.predict_proba(scaled)[0][1]
            )

            # Quantum score from the saved weights.
            q_score = quantum_score(scaled[0])

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Classical model")
                st.metric(
                    "Estimated churn probability",
                    f"{classical_probability:.1%}",
                )
                if classical_probability >= 0.5:
                    st.error("Model prediction: Higher churn risk")
                else:
                    st.success("Model prediction: Lower churn risk")

            with col2:
                st.subheader("Quantum circuit")
                st.metric("Quantum churn score", f"{q_score:.1%}")
                st.caption(
                    "Circuit-derived score; not a calibrated probability."
                )
                if q_score >= 0.5:
                    st.warning("Score is on the higher side")
                else:
                    st.info("Score is on the lower side")

            st.subheader("Customer features used")
            input_table = pd.DataFrame(raw, columns=feature_names)
            st.dataframe(input_table, use_container_width=True)

            # Store session-only history.
            if "prediction_history" not in st.session_state:
                st.session_state.prediction_history = []

            st.session_state.prediction_history.append(
                {
                    "Tenure": tenure,
                    "Internet service": fiber,
                    "Payment method": payment,
                    "Contract": contract,
                    "Classical churn probability": classical_probability,
                    "Quantum churn score": q_score,
                }
            )

        except Exception as error:
            st.error(f"Prediction failed: {error}")

    if st.session_state.get("prediction_history"):
        st.subheader("Current session prediction history")
        history = pd.DataFrame(st.session_state.prediction_history)
        st.dataframe(history, use_container_width=True)

        csv_data = history.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download prediction history (CSV)",
            data=csv_data,
            file_name="churn_prediction_history.csv",
            mime="text/csv",
        )


# --------------------------------------------------
# MODEL COMPARISON
# --------------------------------------------------
elif page == "Model Comparison":
    st.header("Quantum vs Classical Model")

    comparison = pd.DataFrame(
        {
            "Metric": list(quantum_metrics.keys()),
            "Quantum model": list(quantum_metrics.values()),
            "Classical baseline": list(classical_metrics.values()),
        }
    ).set_index("Metric")

    st.dataframe(
        comparison.style.format("{:.2%}"),
        use_container_width=True,
    )

    st.subheader("Metric comparison")
    st.bar_chart(comparison)

    st.markdown(
        """
        **Interpretation**

        - Accuracy measures the fraction of correct predictions.
        - Precision measures how many predicted churners actually churned.
        - Recall measures how many actual churners were detected.
        - F1-score balances precision and recall.
        - ROC-AUC measures ranking ability across thresholds.

        The previously recorded classical baseline outperformed the
        quantum model on accuracy, F1-score, and ROC-AUC. The quantum
        model's training used a small subset of the dataset, so these
        results are preliminary.
        """
    )


# --------------------------------------------------
# EXPLAINABLE AI
# --------------------------------------------------
elif page == "Explainable AI":
    st.header("Explainable AI")

    st.write(
        "SHAP explains the saved classical model. The quantum section "
        "shows a simple sensitivity analysis and is not a SHAP explanation."
    )

    tenure = st.slider(
        "Tenure for explanation (months)",
        0, 72, 12,
        key="shap_tenure",
    )
    fiber = st.selectbox(
        "Internet service",
        ["DSL", "Fiber optic", "No internet service"],
        key="shap_fiber",
    )
    payment = st.selectbox(
        "Payment method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
        key="shap_payment",
    )
    contract = st.selectbox(
        "Contract type",
        ["Month-to-month", "One year", "Two year"],
        key="shap_contract",
    )

    raw = np.array(
        [[
            tenure,
            int(fiber == "Fiber optic"),
            int(payment == "Electronic check"),
            int(contract == "Two year"),
        ]],
        dtype=float,
    )

    scaled = scaler.transform(raw)
    st.subheader("Classical model explanation")

    if shap is None:
        st.warning(
            "SHAP is not installed. Add shap to requirements.txt "
            "and allow the app to redeploy."
        )
    else:
        try:
            background = np.zeros((1, scaled.shape[1]))
            explainer = shap.LinearExplainer(
                classical_model,
                background,
            )
            explanation = explainer(scaled)

            values = np.asarray(explanation.values)

            if values.ndim == 3:
                values = values[0, :, -1]
            elif values.ndim == 2:
                values = values[0]

            if len(values) != len(feature_names):
                raise ValueError(
                    "SHAP returned a shape that does not match "
                    "the saved feature list."
                )

            shap_table = pd.DataFrame(
                {
                    "Feature": feature_names,
                    "SHAP value": values,
                    "Absolute contribution": np.abs(values),
                }
            ).sort_values(
                "Absolute contribution",
                ascending=False,
            )

            st.dataframe(shap_table, use_container_width=True)
            st.bar_chart(
                shap_table.set_index("Feature")[["SHAP value"]]
            )

            st.caption(
                "SHAP values describe contributions on the model's "
                "output scale, generally log-odds for Logistic Regression. "
                "They are not percentage-point changes in probability."
            )

        except Exception as error:
            st.warning(
                "SHAP could not generate the explanation for this model. "
                f"Details: {error}"
            )

    st.divider()
    st.subheader("Quantum feature sensitivity")

    st.caption(
        "This compares the circuit score when one encoded feature is "
        "changed. It is a sensitivity check, not SHAP."
    )

    try:
        baseline = raw[0].copy()
        baseline_scaled = scaler.transform([baseline])[0]
        baseline_score = quantum_score(baseline_scaled)

        sensitivity_rows = []

        for index, name in enumerate(feature_names):
            changed = baseline.copy()

            if name == "tenure":
                changed[index] = 72 if baseline[index] < 36 else 0
            else:
                changed[index] = 1 - changed[index]

            changed_scaled = scaler.transform([changed])[0]
            changed_score = quantum_score(changed_scaled)

            sensitivity_rows.append(
                {
                    "Feature changed": name,
                    "Baseline quantum score": baseline_score,
                    "Changed quantum score": changed_score,
                    "Score difference": changed_score - baseline_score,
                }
            )

        sensitivity = pd.DataFrame(sensitivity_rows)
        st.dataframe(sensitivity, use_container_width=True)

    except Exception as error:
        st.warning(f"Quantum sensitivity analysis failed: {error}")


# --------------------------------------------------
# PROJECT DETAILS
# --------------------------------------------------
elif page == "Project Details":
    st.header("Project Details")

    st.subheader("Project title")
    st.write(
        "Hybrid Quantum–Classical Machine Learning for Telecom "
        "Customer Churn Prediction with Explainable AI"
    )

    st.subheader("Workflow")
    st.code(
        "Telecom data\n"
        "   ↓\n"
        "Preprocessing and feature selection\n"
        "   ↓\n"
        "Four selected features\n"
        "   ↓\n"
        "Four-qubit quantum circuit\n"
        "   ↓\n"
        "Quantum score and classical prediction\n"
        "   ↓\n"
        "SHAP explanation for the classical model",
        language="text",
    )

    st.subheader("Technology stack")
    st.markdown(
        """
        - Python
        - Streamlit
        - Scikit-learn
        - PennyLane quantum simulator
        - SHAP explainability
        - NumPy and Pandas
        """
    )

    st.subheader("Important limitations")
    st.markdown(
        """
        - The quantum circuit runs on a simulator, not physical quantum hardware.
        - The quantum model was trained on a small subset in the preliminary experiment.
        - The quantum score is not guaranteed to be a calibrated probability.
        - SHAP in this app explains the classical model, not the quantum circuit.
        - The preliminary results do not demonstrate quantum advantage.
        """
    )

    st.info(
        "For a final academic report, verify the prediction preprocessing "
        "and circuit implementation against the original training notebook."
    )
