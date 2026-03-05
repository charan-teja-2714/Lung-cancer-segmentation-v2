import io
import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import segmentation_models_pytorch as smp

st.set_page_config(page_title="Lung Cancer Segmentation", layout="wide")

MODEL_PATH = "models/segmentation_multiclass/weights/best_multiclass.pth"
IMAGE_SIZE = 256
MIN_REGION_PX = 150   # minimum predicted region size (pixels, at original image scale)
                      # training masks: smallest real tumor = ~250 px at 512×512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_NAMES = {
    0: "Background",
    1: "ADC (Adenocarcinoma)",
    2: "LCC (Large Cell Carcinoma)",
    3: "SCC (Squamous Cell Carcinoma)"
}

CLASS_COLORS = {
    0: [0, 0, 0],
    1: [255, 0, 0],
    2: [0, 255, 0],
    3: [0, 0, 255]
}

@st.cache_resource
def load_model():
    model = smp.UnetPlusPlus(
        encoder_name="efficientnet-b4",
        encoder_weights=None,
        in_channels=1,
        classes=4
    )
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.to(DEVICE)
    model.eval()
    return model

def to_grayscale(image_np):
    """Convert any image (RGB, RGBA, grayscale) to a uint8 grayscale array."""
    if image_np.ndim == 2:
        return image_np
    if image_np.shape[2] == 4:
        return cv2.cvtColor(image_np, cv2.COLOR_RGBA2GRAY)
    return cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

def preprocess_image(image_np):
    """
    Normalise a CT image to match the training distribution.

    Training images (DICOM-derived PNGs) have a narrow pixel range (0-130)
    with a dark background. Internet images use various window/level settings.
    Percentile clipping (1st-99th) is used instead of a fixed clip so the
    normalization is robust across sources.
    """
    gray = to_grayscale(image_np).astype(np.float32)
    original_size = gray.shape          # (H, W)

    # Robust contrast normalisation
    p_low, p_high = np.percentile(gray, 1), np.percentile(gray, 99)
    if p_high - p_low < 1e-6:          # degenerate image (flat)
        p_low, p_high = gray.min(), gray.max()
    gray = np.clip(gray, p_low, p_high)
    gray = (gray - gray.min()) / (gray.max() - gray.min() + 1e-8)

    image_resized = cv2.resize(gray, (IMAGE_SIZE, IMAGE_SIZE))
    image_tensor = torch.from_numpy(image_resized).unsqueeze(0).unsqueeze(0)
    return image_tensor, original_size

def predict(model, image_tensor):
    with torch.no_grad():
        image_tensor = image_tensor.to(DEVICE)
        output = model(image_tensor)
        pred_mask = torch.argmax(output, dim=1).cpu().numpy()[0]  # (256, 256)
    return pred_mask.astype(np.uint8)

def clean_mask(mask, min_area=MIN_REGION_PX):
    """Remove connected components smaller than min_area pixels."""
    cleaned = mask.copy()
    for class_id in range(1, 4):
        binary = (mask == class_id).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
        for lbl in range(1, num_labels):
            if stats[lbl, cv2.CC_STAT_AREA] < min_area:
                cleaned[labels == lbl] = 0
    return cleaned

def create_colored_mask(mask):
    h, w = mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in CLASS_COLORS.items():
        colored[mask == class_id] = color
    return colored

def create_overlay(image_np, mask, alpha=0.4):
    gray = to_grayscale(image_np)
    image_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    if image_rgb.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(mask, (image_rgb.shape[1], image_rgb.shape[0]),
                          interpolation=cv2.INTER_NEAREST)
    colored = create_colored_mask(mask)
    return cv2.addWeighted(image_rgb, 1 - alpha, colored, alpha, 0)

def calculate_statistics(mask):
    total = mask.size
    stats = {}
    for class_id, class_name in CLASS_NAMES.items():
        count = int(np.sum(mask == class_id))
        stats[class_name] = {
            "pixels": count,
            "percentage": round((count / total) * 100, 2)
        }
    return stats

def to_png_bytes(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()

# ── UI ──────────────────────────────────────────────────────────────────────
st.title("Lung Cancer Segmentation")

with st.sidebar:
    st.markdown("### Class Legend")
    st.markdown("🔴 **Red** — ADC (Adenocarcinoma)")
    st.markdown("🟢 **Green** — LCC (Large Cell Carcinoma)")
    st.markdown("🔵 **Blue** — SCC (Squamous Cell Carcinoma)")
    st.markdown("⬛ **Black** — Background")

uploaded_file = st.file_uploader(
    "Upload a CT scan image (PNG / JPG)",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_np = np.array(image)

    with st.spinner("Running segmentation…"):
        model = load_model()
        image_tensor, original_size = preprocess_image(image_np)
        pred_mask_256 = predict(model, image_tensor)         # 256×256

        # Clean at model output resolution (consistent scale, independent of
        # the uploaded image size). MIN_REGION_PX is calibrated at this scale.
        pred_mask_clean_256 = clean_mask(pred_mask_256, min_area=MIN_REGION_PX)

        # Resize cleaned mask back to original image size
        pred_mask_final = cv2.resize(
            pred_mask_clean_256,
            (original_size[1], original_size[0]),
            interpolation=cv2.INTER_NEAREST
        )

        colored_mask = create_colored_mask(pred_mask_final)
        overlay = create_overlay(image_np, pred_mask_final)

    # ── Images ──
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Original CT Scan**")
        st.image(image, use_column_width=True)
    with col2:
        st.markdown("**Segmentation Overlay**")
        st.image(overlay, use_column_width=True)
    with col3:
        st.markdown("**Predicted Mask**")
        st.image(colored_mask, use_column_width=True)

    # ── Statistics ──
    st.markdown("---")
    stats = calculate_statistics(pred_mask_final)
    detected = [name for name, d in stats.items()
                if name != "Background" and d["percentage"] > 0.1]

    if detected:
        st.error(f"Cancer detected: {', '.join(detected)}")
    else:
        st.success("No significant cancer regions detected")
        st.caption(
            "If you expected a detection, ensure the image is an axial CT scan "
            "slice (not a chest X-ray or MRI) with a visible lung cross-section."
        )

    metric_cols = st.columns(3)
    cancer_classes = [(n, d) for n, d in stats.items() if n != "Background"]
    for col, (class_name, data) in zip(metric_cols, cancer_classes):
        col.metric(label=class_name, value=f"{data['percentage']}%",
                   delta=f"{data['pixels']:,} px")

    # ── Downloads ──
    st.markdown("---")
    dl1, dl2 = st.columns(2)
    with dl1:
        st.download_button("Download Overlay", data=to_png_bytes(overlay),
                           file_name="overlay.png", mime="image/png")
    with dl2:
        st.download_button("Download Mask", data=to_png_bytes(colored_mask),
                           file_name="mask.png", mime="image/png")

else:
    st.info("Upload a CT scan image above to begin.")
