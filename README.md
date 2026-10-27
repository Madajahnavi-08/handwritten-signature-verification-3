# Handwritten Signature Verification

**Student project:** A Siamese Neural Network-Based Approach to Handwritten Signature Verification  
**Author:** Mada Jahnavi  
**Domain:** Deep Learning and Computer Vision

## Important project note

The included Streamlit app is a **pixel-similarity prototype**. It is not a trained Siamese Neural Network and must not be used for bank authentication. The reference accuracy graph is illustrative only, not a measurement from this app.

## Files

- `app.py` — Streamlit app for comparing uploaded signature images, displaying pixel similarity and comparison time, and calculating evaluation metrics from labelled CSV predictions.
- `requirements.txt` — Python dependencies.
- `data/README.md` — Instructions for adding your own properly labelled test dataset.

## Run locally

1. Install Python 3.10 or newer.
2. Open a terminal in this project folder.
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Start the app:

   ```bash
   streamlit run app.py
   ```

## Evaluation CSV format

Upload a CSV with exactly these columns:

```csv
actual,predicted
1,1
1,0
0,0
0,0
```

- `1` = genuine
- `0` = forged
- `actual` = ground-truth label
- `predicted` = prediction made by the method being evaluated

The rows above are format examples only. Do not use them as real evaluation results. Use a properly labelled test dataset and predictions from the method being evaluated.

## Deploy to Streamlit Community Cloud

1. Upload `app.py`, `requirements.txt`, `README.md`, and the `data` folder to the root of a GitHub repository.
2. Go to https://share.streamlit.io/
3. Sign in with GitHub and select **Create app**.
4. Choose your repository, branch `main`, and main file path `app.py`.
5. Click **Deploy**.

## Limitations

Pixel similarity can be affected by image alignment, size, background, and writing variation. A genuine Siamese Neural Network requires a training dataset, training code, saved model, and evaluation on a separate test set. Do not claim Siamese-network results until that model has been trained and tested.
