import pandas as pd
import streamlit as st
import numpy as np
import pickle
import os
from tensorflow.keras.models import load_model
from PIL import Image


# Inline small helpers (merged to reduce separate files)
IMG_SIZE = 128

def preprocess_image(img_file):
    try:
        if isinstance(img_file, Image.Image):
            img = img_file
        else:
            img = Image.open(img_file)
        if img.mode != 'L':
            img = img.convert('L')
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = np.expand_dims(arr, axis=-1)
        arr = np.expand_dims(arr, axis=0)
        return arr
    except Exception as e:
        raise ValueError(f"Error processing image: {str(e)}")


def interaction_toxicity(prob1, prob2, mlp_prob):
    final_prob = 0.2 * prob1 + 0.2 * prob2 + 0.6 * mlp_prob
    toxicity_percent = final_prob * 100
    label = "Toxic" if final_prob > 0.5 else "Non Toxic"
    return toxicity_percent, label


# ==============================
# LOAD MODELS & SCALER
# ==============================

error_msg = None
try:
    cnn_model = load_model("models/cnn_model.h5")
    mlp_model = load_model("models/mlp_model.h5")
    scaler = pickle.load(open("scaler/scaler.pkl", "rb"))
    np_encoder = pickle.load(open("scaler/np_encoder.pkl", "rb"))
    class_encoder = pickle.load(open("scaler/class_encoder.pkl", "rb"))
    models_loaded = True
except Exception as e:
    error_msg = str(e)
    models_loaded = False

models_dict = {
    'loaded': models_loaded,
    'error': error_msg,
    'cnn_model': cnn_model if models_loaded else None,
    'mlp_model': mlp_model if models_loaded else None,
    'scaler': scaler if models_loaded else None,
    'np_encoder': np_encoder if models_loaded else None,
    'class_encoder': class_encoder if models_loaded else None
}

# ==============================
# UI - Minimal: images + 9 features
# ==============================

st.title("🧪 Nano-Particle Interaction Toxicity Predictor")
st.markdown("Upload two images and enter 9 bio-features. The app predicts toxicity % and classification.")

if not models_dict["loaded"]:
    st.error(f"❌ Error loading models: {models_dict['error']}")
    st.error("Please ensure all model files are in the correct directories")
    st.stop()


left_col, right_col = st.columns([1, 1])
with left_col:
    img1 = st.file_uploader("Upload Nano Image 1", type=['png','jpg','jpeg','bmp','tif','tiff'], key="img1")
    img2 = st.file_uploader("Upload Nano Image 2", type=['png','jpg','jpeg','bmp','tif','tiff'], key="img2")
    if img1 and img2:
        st.image([img1, img2], width=300)

with right_col:
    st.subheader("Enter Bio-Properties (manual)")
    np_type = st.selectbox("Nanoparticle Type", models_dict["np_encoder"].classes_)
    coresize   = st.number_input("Core Size", value=10.0)
    hydrosize  = st.number_input("Hydro Size", value=15.0)
    surfcharge = st.number_input("Surface Charge", value=0.0)
    surfarea   = st.number_input("Surface Area", value=100.0)
    ec         = st.number_input("Ec", value=0.0)
    expotime   = st.number_input("Exposure Time", value=24.0)
    dosage     = st.number_input("Dosage", value=10.0)
    e_val      = st.number_input("e", value=1.0)
    noxygen    = st.number_input("NOxygen", value=5.0)
    run_predict = st.button("Run Prediction")
    expected_label = st.selectbox("(Optional) Expected Classification", ["Unknown","Non Toxic","Toxic"], index=0)

def _extract_prob(pred):
    arr = np.array(pred)
    if arr.size == 0:
        return 0.0
    if arr.ndim == 1:
        if arr.shape[0] == 1:
            return float(arr[0])
        else:
            return float(arr[1])
    if arr.ndim == 2:
        if arr.shape[1] == 1:
            return float(arr[0,0])
        else:
            return float(arr[0,1])
    return float(arr.flatten()[-1])

def _mlp_input_dim(model):
    try:
        shape = model.input_shape
        if isinstance(shape, list):
            shape = shape[0]
        if isinstance(shape, tuple):
            return int(shape[-1])
        return int(shape)
    except Exception:
        try:
            first = model.layers[0]
            if hasattr(first, 'input_shape') and first.input_shape:
                return int(first.input_shape[-1])
        except Exception:
            return None

if img1 and img2 and run_predict:
    try:
        p1 = preprocess_image(img1)
        p2 = preprocess_image(img2)
        pred1 = models_dict["cnn_model"].predict(p1, verbose=0)
        pred2 = models_dict["cnn_model"].predict(p2, verbose=0)
        prob1 = _extract_prob(pred1)
        prob2 = _extract_prob(pred2)

        cols = ['NPs','coresize','hydrosize','surfcharge','surfarea','Ec','Expotime','dosage','e','NOxygen']
        try:
            np_encoded = models_dict["np_encoder"].transform([np_type])[0]
        except Exception:
            np_encoded = 0

        bio_df = pd.DataFrame([{
            'NPs': np_encoded,
            'coresize': coresize,
            'hydrosize': hydrosize,
            'surfcharge': surfcharge,
            'surfarea': surfarea,
            'Ec': ec,
            'Expotime': expotime,
            'dosage': dosage,
            'e': e_val,
            'NOxygen': noxygen
        }], columns=cols)

        bio_scaled = models_dict["scaler"].transform(bio_df)
        # Determine how many features the MLP expects and slice accordingly
        n_expected = _mlp_input_dim(models_dict["mlp_model"])
        if n_expected is None:
            bio_scaled_mlp = bio_scaled
        else:
            if bio_scaled.shape[1] < n_expected:
                raise ValueError(f"MLP expects {n_expected} features but scaler produced {bio_scaled.shape[1]}")
            bio_scaled_mlp = bio_scaled[:, :n_expected]
        pred_mlp = models_dict["mlp_model"].predict(bio_scaled_mlp, verbose=0)
        mlp_prob = _extract_prob(pred_mlp)

        toxicity_percent, label = interaction_toxicity(prob1, prob2, mlp_prob)

        # Map toxicity percent to risk category
        def risk_category(pct: float):
            if pct <= 25:
                return "Very Low"
            if pct <= 50:
                return "Low-Moderate"
            if pct <= 75:
                return "Moderate-High"
            return "Very High"

        risk = risk_category(toxicity_percent)

        with st.container():
            st.subheader("Results")
            rcol1, rcol2 = st.columns([1, 2])
            with rcol1:
                st.metric("Interaction Toxicity %", f"{toxicity_percent:.2f}%")
                st.progress(min(max(toxicity_percent/100.0, 0.0), 1.0))
            with rcol2:
                if label == 'Toxic':
                    st.error(f"⚠️ Classification: {label}")
                else:
                    st.success(f"✅ Classification: {label}")
                st.write(f"**Risk category:** {risk}")

        st.markdown("---")
        col_a, col_b, col_c = st.columns(3)
        col_a.write("Image 1 CNN Score:")
        col_a.write(f"{prob1:.4f}")
        col_b.write("Image 2 CNN Score:")
        col_b.write(f"{prob2:.4f}")
        col_c.write("Bio-Feature MLP Score:")
        col_c.write(f"{mlp_prob:.4f}")

        # Compare with expected label if provided
        if expected_label != "Unknown":
            match = (expected_label == 'Toxic' and label == 'Toxic') or (expected_label == 'Non Toxic' and label == 'Non Toxic')
            if match:
                st.success(f"✅ Matches expected: {expected_label}")
            else:
                st.warning(f"❗ Does not match expected ({expected_label})")

        # Debug expander with raw arrays for troubleshooting
        with st.expander("Debug: raw model outputs and shapes"):
            st.write({
                'pred1_raw': getattr(pred1, 'tolist', lambda: pred1)(),
                'pred2_raw': getattr(pred2, 'tolist', lambda: pred2)(),
                'pred_mlp_raw': getattr(pred_mlp, 'tolist', lambda: pred_mlp)(),
                'prob1': prob1,
                'prob2': prob2,
                'mlp_prob': mlp_prob,
                'bio_scaled_shape': bio_scaled.shape
            })

    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        import traceback
        st.text(traceback.format_exc())
elif not run_predict:
    st.info("Fill inputs and click 'Run Prediction' to compute results")
else:
    st.info("Please upload both images to run prediction")
