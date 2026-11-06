
import time
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

# ---------------- PAGE CONFIGURATION ----------------

st.set_page_config(
    page_title="Handwritten Signature Verification",
    page_icon="✍️",
    layout="wide"
)

st.title("✍️ Handwritten Signature Verification")
st.subheader("B.Tech Mini Project — Deep Learning & Computer Vision")
st.caption("Developed by Mada Jahnavi")

st.warning(
    "Educational prototype: Current image comparison uses pixel similarity. "
    "It is not a trained Siamese Neural Network or a banking authentication system."
)

# ---------------- IMAGE PROCESSING ----------------

def prepare_image(uploaded_file):
    image = Image.open(uploaded_file).convert("L")
    image = ImageOps.autocontrast(image)
    image = ImageOps.pad(image, (220, 155), color=255)
    return image


def compare_images(img1, img2):
    a = np.asarray(img1, dtype=np.float32) / 255.0
    b = np.asarray(img2, dtype=np.float32) / 255.0

    score = (1.0 - np.mean(np.abs(a - b))) * 100

    return float(np.clip(score, 0, 100))


# ====================================================
# SECTION 1: SIGNATURE COMPARISON
# ====================================================

st.header("1. Compare Two Signatures")

left, right = st.columns(2)

with left:
    reference_file = st.file_uploader(
        "Upload reference signature",
        type=["png", "jpg", "jpeg"],
        key="reference"
    )

with right:
    test_file = st.file_uploader(
        "Upload signature to compare",
        type=["png", "jpg", "jpeg"],
        key="test"
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

            st.divider()

            st.subheader("Comparison Results")

            m1, m2 = st.columns(2)

            m1.metric("Pixel Similarity", f"{score:.2f}%")

            m2.metric(
                "Comparison Time",
                f"{elapsed:.6f} seconds"
            )

            st.progress(int(round(score)))

            st.subheader("Signature Verification Result")

            # Illustrative threshold only.
            # Calibrate using labelled validation data.
            THRESHOLD = 80.0

            if score >= THRESHOLD:

                st.success("HIGH SIMILARITY MATCH")

                st.write(
                    "The uploaded signature has high pixel similarity "
                    "with the reference signature."
                )

            else:

                st.error("LOW SIMILARITY MATCH")

                st.write(
                    "The uploaded signature has low pixel similarity "
                    "with the reference signature."
                )

            st.info(
                "This is a preliminary pixel-based comparison. "
                "It does not confirm whether a signature is genuine or forged."
            )

    except Exception as error:

        st.error(
            f"Could not process uploaded images: {error}"
        )


# ====================================================
# SECTION 2: EVALUATION USING CSV
# ====================================================

st.divider()

st.header("2. Evaluate Labelled Test Results")

st.write(
    "Upload a CSV containing actual and predicted labels "
    "from your labelled test dataset."
)

st.write(
    "**Labels:** 1 = Genuine, 0 = Forged"
)

st.code(
    "actual,predicted\n"
    "1,1\n"
    "1,0\n"
    "0,0\n"
    "0,1",
    language="csv"
)

st.caption(
    "Example format only. Upload real test-set results for evaluation."
)

csv_file = st.file_uploader(
    "Upload Evaluation CSV",
    type=["csv"],
    key="evaluation_csv"
)

if csv_file is not None:

    try:

        results = pd.read_csv(csv_file)

        results.columns = results.columns.str.strip()

        required = {"actual", "predicted"}

        if not required.issubset(results.columns):

            st.error(
                "CSV must contain actual and predicted columns."
            )

        elif results.empty:

            st.error("CSV contains no test records.")

        else:

            actual_values = pd.to_numeric(
                results["actual"],
                errors="coerce"
            )

            predicted_values = pd.to_numeric(
                results["predicted"],
                errors="coerce"
            )

            if actual_values.isna().any() or predicted_values.isna().any():

                st.error("CSV contains missing or invalid labels.")

            elif not set(actual_values.unique()).issubset({0, 1}):

                st.error("Actual labels must be 0 or 1.")

            elif not set(predicted_values.unique()).issubset({0, 1}):

                st.error("Predicted labels must be 0 or 1.")

            else:

                actual = actual_values.astype(int).to_numpy()
                predicted = predicted_values.astype(int).to_numpy()

                # Confusion matrix values

                tp = int(np.sum((actual == 1) & (predicted == 1)))
                tn = int(np.sum((actual == 0) & (predicted == 0)))
                fp = int(np.sum((actual == 0) & (predicted == 1)))
                fn = int(np.sum((actual == 1) & (predicted == 0)))

                total = len(actual)

                accuracy = (tp + tn) / total * 100

                precision = (
                    tp / (tp + fp) * 100
                    if tp + fp else 0.0
                )

                recall = (
                    tp / (tp + fn) * 100
                    if tp + fn else 0.0
                )

                f1 = (
                    2 * precision * recall / (precision + recall)
                    if precision + recall else 0.0
                )

                error_rate = 100 - accuracy

                # Display evaluation metrics

                st.subheader("Measured Evaluation Metrics")

                c1, c2, c3, c4 = st.columns(4)

                c1.metric("Accuracy", f"{accuracy:.2f}%")
                c2.metric("Precision", f"{precision:.2f}%")
                c3.metric("Recall", f"{recall:.2f}%")
                c4.metric("F1-score", f"{f1:.2f}%")

                st.metric("Error Rate", f"{error_rate:.2f}%")

                # Confusion matrix

                st.subheader("Confusion Matrix")

                matrix = pd.DataFrame(
                    [[tn, fp], [fn, tp]],
                    index=[
                        "Actual Forged (0)",
                        "Actual Genuine (1)"
                    ],
                    columns=[
                        "Predicted Forged (0)",
                        "Predicted Genuine (1)"
                    ]
                )

                st.dataframe(
                    matrix,
                    use_container_width=True
                )

                # Performance graph

                st.subheader("Performance Analysis")

                names = [
                    "Accuracy",
                    "Precision",
                    "Recall",
                    "F1-score"
                ]

                values = [
                    accuracy,
                    precision,
                    recall,
                    f1
                ]

                fig, ax = plt.subplots(figsize=(9, 4))

                bars = ax.bar(names, values)

                ax.set_ylim(0, 100)
                ax.set_ylabel("Percentage (%)")
                ax.set_title("Model Evaluation Performance")

                ax.bar_label(
                    bars,
                    fmt="%.2f%%"
                )

                fig.tight_layout()

                st.pyplot(fig)

                plt.close(fig)

                st.success(
                    f"Evaluation completed using {total} test records."
                )

    except Exception as error:

        st.error(f"CSV evaluation error: {error}")


# ====================================================
# SECTION 3: PERFORMANCE SUMMARY
# ====================================================

st.divider()

st.header("3. Performance Summary")

st.write(
    "The performance metrics above are calculated from the "
    "uploaded labelled test dataset."
)

st.info(
    "For a valid comparison between an existing method and the "
    "proposed Siamese Neural Network, evaluate both methods on "
    "the same test dataset. Do not enter assumed accuracy values."
)

st.caption(
    "Project: Siamese Neural Network-Based Handwritten Signature Verification"
)

st.caption("Developed by Mada Jahnavi")
