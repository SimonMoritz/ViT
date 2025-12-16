import streamlit as st
from PIL import Image, ImageDraw
from pathlib import Path

IMG_DIR = Path("Airport_Dataset_v0_images")
LBL_DIR = Path("Airport_Dataset_v0_labels")

images = sorted(IMG_DIR.glob("*.jpg"))
assert images, "No images found"

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
            cls, xc, yc, bw, bh = map(float, line.split())

            x1 = (xc - bw / 2) * w
            y1 = (yc - bh / 2) * h
            x2 = (xc + bw / 2) * w
            y2 = (yc + bh / 2) * h

            draw.rectangle([x1, y1, x2, y2], outline="lime", width=3)

st.image(
    img,
    caption=f"{img_path.name} ({st.session_state.idx + 1}/{len(images)})",
    use_container_width=True,
)

st.caption("Buttons and slider are now fully in sync")
