import time
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

st.set_page_config(
    page_title="Handwritten Signature Verification",
    page_icon="✍️",
    layout="wide"
)

st.title("✍️ Handwritten Signature Verification")
st.subheader("B.Tech Mini Project — Deep Learning & Computer Vision")
st.caption("Developed by Mada Jahnavi")

st.warning(
    "Educational demo only. This app uses pixel similarity, not a trained "
    "Siamese Neural Network, and must not be used for bank authentication."
)

def prepare_image(uploaded_file):
    image = Image.open(uploaded_file).convert("L")
    image = ImageOps.autocontrast(image)
    return ImageOps.pad(image, (220, 155), color=255)

def compare_images(img1, img2):
    a = np.asarray(img1, dtype=np.float32) / 255.0
    b = np.asarray(img2, dtype=np.float32) / 255.0
    score = (1.0 - np.mean(np.abs(a - b))) * 100.0
    return float(np.clip(score, 0, 100))

st.header("1. Compare Two Signatures")
left, right = st.columns(2)

with left:
    reference_file = st.file_uploader(
        "Upload reference signature", type=["png", "jpg", "jpeg"], key="reference"
    )
with right:
    test_file = st.file_uploader(
        "Upload signature to compare", type=["png", "jpg", "jpeg"], key="test"
    )

if reference_file is not None and test_file is not None:
    try:
        image1 = prepare_image(reference_file)
        image2 = prepare_image(test_file)
        col1, col2 = st.columns(2)
        with col1:
            st.image(image1, caption="Reference Signature")
        with col2:
            st.image(image2, caption="Signature to Compare")

        if st.button("Compare Signatures", type="primary"):
            start = time.perf_counter()
            score = compare_images(image1, image2)
            elapsed = time.perf_counter() - start
            m1, m2 = st.columns(2)
            m1.metric("Pixel Similarity", f"{score:.2f}%")
            m2.metric("Comparison Time", f"{elapsed:.6f} seconds")
            st.progress(int(round(score)))
            st.info(
                "Similarity is not verification accuracy. This score alone "
                "cannot determine whether a signature is genuine or forged."
            )
    except Exception as error:
        st.error(f"Could not process the uploaded images: {error}")

st.divider()
st.header("2. Evaluate Labelled Test Results")
st.write(
    "Upload a CSV with actual and predicted labels from your labelled test set. "
    "Use 1 for genuine and 0 for forged."
)
st.code(
    "actual,predicted\n"
    "1,1\n"
    "1,0\n"
    "0,0\n"
    "0,0",
    language="csv"
)
st.caption("These rows only demonstrate the file format; do not use them as real results.")

csv_file = st.file_uploader("Upload evaluation CSV", type=["csv"], key="evaluation_csv")

if csv_file is not None:
    try:
        results = pd.read_csv(csv_file)
        required = {"actual", "predicted"}
        if not required.issubset(results.columns):
            st.error("CSV must contain columns named actual and predicted.")
        elif results.empty:
            st.error("The CSV contains no test records.")
        elif results[["actual", "predicted"]].isna().any().any():
            st.error("Remove missing labels from the CSV.")
        elif not set(results["actual"].unique()).issubset({0, 1}) or not set(
            results["predicted"].unique()
        ).issubset({0, 1}):
            st.error("Labels must be 0 (forged) or 1 (genuine).")
        else:
            actual = results["actual"].astype(int).to_numpy()
            predicted = results["predicted"].astype(int).to_numpy()
            tp = int(np.sum((actual == 1) & (predicted == 1)))
            tn = int(np.sum((actual == 0) & (predicted == 0)))
            fp = int(np.sum((actual == 0) & (predicted == 1)))
            fn = int(np.sum((actual == 1) & (predicted == 0)))

            accuracy = (tp + tn) / len(actual) * 100
            precision = tp / (tp + fp) * 100 if tp + fp else 0.0
            recall = tp / (tp + fn) * 100 if tp + fn else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if precision + recall else 0.0
            )

            st.subheader("Measured Evaluation Metrics")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy", f"{accuracy:.2f}%")
            c2.metric("Precision", f"{precision:.2f}%")
            c3.metric("Recall", f"{recall:.2f}%")
            c4.metric("F1-score", f"{f1:.2f}%")

            st.write("Confusion matrix counts")
            st.dataframe(
                pd.DataFrame(
                    [[tn, fp], [fn, tp]],
                    index=["Actual Forged (0)", "Actual Genuine (1)"],
                    columns=["Predicted Forged (0)", "Predicted Genuine (1)"]
                ),
                use_container_width=True
            )

            fig, ax = plt.subplots()
            names = ["Accuracy", "Precision", "Recall", "F1-score"]
            values = [accuracy, precision, recall, f1]
            bars = ax.bar(names, values)
            ax.set_ylim(0, 100)
            ax.set_ylabel("Score (%)")
            ax.set_title("Measured Test Dataset Performance")
            ax.bar_label(bars, fmt="%.2f%%")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            st.caption(f"Calculated from {len(actual)} labelled test pairs.")
    except Exception as error:
        st.error(f"Could not evaluate the CSV: {error}")

st.divider()
st.header("3. Published Reference Accuracy (Illustration Only)")
st.write(
    "These values are reference examples, not measurements from this application. "
    "Replace them with results from a verified source before including them in a report."
)
methods = ["Existing SVM (reference)", "Siamese NN (reference)"]
values = [76.57, 90.10]
fig, ax = plt.subplots()
bars = ax.bar(methods, values)
ax.set_ylim(0, 100)
ax.set_ylabel("Accuracy (%)")
ax.set_title("Reference Accuracy Comparison — Not Project Measurements")
ax.tick_params(axis="x", labelrotation=10)
ax.bar_label(bars, fmt="%.2f%%")
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)
