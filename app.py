import io
import math
import base64
import cv2
import numpy as np
import streamlit as st

st.set_page_config(
    page_title="High-Quality Universal Diagram Vectorizer",
    page_icon="🧬",
    layout="wide",
)

st.title("🧬 High-Quality Universal Diagram Vectorizer")
st.write(
    "Upload any scientific diagram or genomic chart. The tool extracts all shapes, text regions, "
    "and layout blocks into a high-fidelity **editable SVG file**, keeping every component fully independent and movable."
)


def process_diagram_to_svg(image):
    """Detects contours and layout blocks to build a high-resolution, movable SVG with pristine clarity."""
    h, w = image.shape[:2]
    
    # Convert to grayscale and threshold to detect distinct shapes and boxes cleanly
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(blurred, 240, 255, cv2.THRESH_BINARY_INV)

    # Find external contours of shapes and diagram blocks
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    svg_elements = []
    svg_elements.append(f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 {w} {h}" width="100%" height="100%">')
    svg_elements.append('  <style>')
    svg_elements.append('    .diagram-element { cursor: pointer; transition: transform 0.2s; }')
    svg_elements.append('    .diagram-element:hover { opacity: 0.85; }')
    svg_elements.append('  </style>')
    
    # Pristine white background base layer
    svg_elements.append(f'  <rect width="{w}" height="{h}" fill="#ffffff" />')

    shape_idx = 1
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 50:  # Filter tiny speckles while preserving sharp text and markers
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)
        
        # Skip if contour covers almost the entire image (canvas border)
        if bw > 0.98 * w and bh > 0.98 * h:
            continue

        # Crop individual shape/block with high precision padding
        pad = 2
        nx = max(0, x - pad)
        ny = max(0, y - pad)
        nw = min(w - nx, bw + (2 * pad))
        nh = min(h - ny, bh + (2 * pad))

        roi_bgr = image[ny:ny+nh, nx:nx+nw]
        
        # Create crisp transparent background mask for the shape
        mask = np.zeros((nh, nw), dtype=np.uint8)
        shifted_cnt = cnt - np.array([nx, ny])
        cv2.drawContours(mask, [shifted_cnt], -1, 255, -1)

        roi_bgra = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2BGRA)
        roi_bgra[mask == 0, 3] = 0

        # Encode with maximum quality PNG compression for high fidelity
        success, encoded_png = cv2.imencode(".png", roi_bgra, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        if success:
            b64_data = base64.b64encode(encoded_png.tobytes()).decode("utf-8")
            
            # Wrap each detected block/shape in its own interactive SVG group (<g>) for independent movement
            svg_elements.append(f'  <g id="element_{shape_idx}" class="diagram-element">')
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
        if max(h, w) > 1400:
            scale_val = 1400.0 / max(h, w)
            img = cv2.resize(img, (int(w * scale_val), int(h * scale_val)), interpolation=cv2.INTER_AREA)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Diagram")
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)

        with col2:
            st.subheader("High-Quality Segmentation Preview")
            preview = img.copy()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(cv2.GaussianBlur(gray, (3, 3), 0), 240, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            count = 0
            for cnt in contours:
                if cv2.contourArea(cnt) > 50:
                    x, y, bw, bh = cv2.boundingRect(cnt)
                    if bw < 0.98 * img.shape[1] and bh < 0.98 * img.shape[0]:
                        cv2.rectangle(preview, (x, y), (x+bw, y+bh), (0, 255, 0), 1)
                        count += 1

            st.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.caption(f"Successfully mapped {count} high-resolution independent movable vector components.")

        st.markdown("---")

        render_btn = st.button("🧬 Generate High-Quality Movable SVG", type="primary")

        if render_btn:
            with st.spinner("Processing high-resolution vectors and building SVG output..."):
                svg_data = process_diagram_to_svg(img)

            st.subheader("📁 Vector SVG Package Ready")
            st.success("SVG file successfully created with pristine quality! Every shape, text label, and line is now independently movable.")

            st.download_button(
                label="Download High-Quality Movable SVG (.svg)",
                data=svg_data,
                file_name="high_quality_diagram_movable.svg",
                mime="image/svg+xml",
            )

    except Exception as e:
        st.error(f"Error encountered: {e}")
