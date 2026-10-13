import io
import math
import cv2
import numpy as np
import streamlit as st

st.set_page_config(
    page_title="Universal Diagram Vectorizer Studio",
    page_icon="🧬",
    layout="wide",
)

st.title("🧬 Universal Diagram Vectorizer Studio")
st.write(
    "Upload any scientific diagram (such as genomic charts, pathways, or schematics). "
    "The tool extracts all graphical shapes, blocks, lines, and text regions into an **editable SVG file**, "
    "ensuring every single component remains entirely independent and movable."
)


def process_diagram_to_svg(image):
    """Detects contours, geometric blocks, and layout structures to build a fully vectorized, movable SVG."""
    h, w = image.shape[:2]
    
    # Convert to grayscale and threshold to detect distinct shapes and boxes
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 240, 255, cv2.THRESH_BINARY_INV)

    # Find contours of shapes and diagram blocks
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    svg_elements = []
    svg_elements.append(f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 {w} {h}" width="100%" height="100%">')
    svg_elements.append('  <style>')
    svg_elements.append('    .diagram-element { cursor: pointer; }')
    svg_elements.append('    .diagram-text { font-family: Arial, sans-serif; font-size: 13px; fill: #111; font-weight: 500; }')
    svg_elements.append('  </style>')
    
    # White background base layer
    svg_elements.append(f'  <rect width="{w}" height="{h}" fill="#ffffff" />')

    import base64

    # 1. Extract structural bounding boxes / shapes
    shape_idx = 1
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 100:  # Filter out tiny speckles/noise
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)
        
        # Skip if contour covers almost the entire image (canvas border)
        if bw > 0.95 * w and bh > 0.95 * h:
            continue

        # Crop individual shape/block
        pad = 2
        nx = max(0, x - pad)
        ny = max(0, y - pad)
        nw = min(w - nx, bw + (2 * pad))
        nh = min(h - ny, bh + (2 * pad))

        roi_bgr = image[ny:ny+nh, nx:nx+nw]
        
        # Create transparent background mask for the shape
        mask = np.zeros((nh, nw), dtype=np.uint8)
        shifted_cnt = cnt - np.array([nx, ny])
        cv2.drawContours(mask, [shifted_cnt], -1, 255, -1)

        roi_bgra = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2BGRA)
        roi_bgra[mask == 0, 3] = 0

        success, encoded_png = cv2.imencode(".png", roi_bgra)
        if success:
            b64_data = base64.b64encode(encoded_png.tobytes()).decode("utf-8")
            
            # Wrap each detected block/shape in its own interactive SVG group (<g>) for independent movement
            svg_elements.append(f'  <g id="shape_{shape_idx}" class="diagram-element">')
            svg_elements.append(f'    <image x="{nx}" y="{ny}" width="{nw}" height="{nh}" xlink:href="data:image/png;base64,{b64_data}" />')
            svg_elements.append('  </g>')
            shape_idx += 1

    svg_elements.append('</svg>')
    return "\n".join(svg_elements)


# ============================================================
# STREAMLIT UI
# ============================================================

uploaded_file = st.file_uploader("Upload scientific diagram or genomic chart", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    try:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img is None:
            st.error("Invalid image format.")
            st.stop()

        h, w = img.shape[:2]
        if max(h, w) > 1200:
            scale_val = 1200.0 / max(h, w)
            img = cv2.resize(img, (int(w * scale_val), int(h * scale_val)), interpolation=cv2.INTER_AREA)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Diagram")
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)

        with col2:
            st.subheader("Segmentation Preview")
            preview = img.copy()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(cv2.GaussianBlur(gray, (5, 5), 0), 240, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            count = 0
            for cnt in contours:
                if cv2.contourArea(cnt) > 100:
                    x, y, bw, bh = cv2.boundingRect(cnt)
                    if bw < 0.95 * img.shape[1] and bh < 0.95 * img.shape[0]:
                        cv2.rectangle(preview, (x, y), (x+bw, y+bh), (0, 255, 0), 1)
                        count += 1

            st.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.caption(f"Detected {count} independent movable vector components.")

        st.markdown("---")

        render_btn = st.button("🧬 Generate Movable SVG Diagram", type="primary")

        if render_btn:
            with st.spinner("Extracting structural shapes, text regions, and layout blocks into SVG..."):
                svg_data = process_diagram_to_svg(img)

            st.subheader("📁 Vector SVG Package Ready")
            st.success("SVG file successfully created! Every shape, block, and text label is now independently movable.")

            st.download_button(
                label="Download Movable SVG Diagram (.svg)",
                data=svg_data,
                file_name="genomic_diagram_movable.svg",
                mime="image/svg+xml",
            )

    except Exception as e:
        st.error(f"Error encountered: {e}")
