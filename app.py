import streamlit as st
import torch
from PIL import Image, ImageDraw

from sar.config import RAW_IMAGE_DIR, RAW_LABEL_DIR
from sar.utils.boxes import box_cxcywh_to_xyxy

IMG_DIR = RAW_IMAGE_DIR
LBL_DIR = RAW_LABEL_DIR

images = sorted(IMG_DIR.glob("*.jpg"))
if not images:
    raise ValueError(f"No images found in {IMG_DIR}")

# ---------------- session state ----------------
if "idx" not in st.session_state:
    st.session_state.idx = 0


def next_img():
    st.session_state.idx = min(st.session_state.idx + 1, len(images) - 1)


def prev_img():
    st.session_state.idx = max(st.session_state.idx - 1, 0)


# ----------------------------------------------

# --- navigation UI ---
col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    st.button("⬅ Previous", on_click=prev_img)
with col3:
    st.button("Next ➡", on_click=next_img)

# 🔑 slider bound directly to session_state.idx
st.slider(
    "Image index",
    0,
    len(images) - 1,
    key="idx",
)

# ---------------- image ----------------
img_path = images[st.session_state.idx]
label_path = LBL_DIR / (img_path.stem + ".txt")

img = Image.open(img_path).convert("RGB")
draw = ImageDraw.Draw(img)

w, h = img.size

if label_path.exists():
    with open(label_path) as f:
        for line in f:
            _cls, xc, yc, bw, bh = map(float, line.split())
            box = box_cxcywh_to_xyxy(torch.tensor([[xc, yc, bw, bh]], dtype=torch.float32))[0]
            x1, y1, x2, y2 = box.tolist()
            x1 *= w
            y1 *= h
            x2 *= w
            y2 *= h

            draw.rectangle([x1, y1, x2, y2], outline="lime", width=3)

st.image(
    img,
    caption=f"{img_path.name} ({st.session_state.idx + 1}/{len(images)})",
    use_container_width=True,
)

st.caption("Buttons and slider are now fully in sync")
