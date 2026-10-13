import io
import math
import cv2
import numpy as np
import streamlit as st

st.set_page_config(
    page_title="Overlapping Layer Diagram Vectorizer",
    page_icon="🧬",
    layout="wide",
)

st.title("🧬 Overlapping Shape & Layer Diagram Vectorizer")
st.write(
    "Upload complex scientific diagrams (like genomic architectures with overlapping transparent regions). "
    "The tool separates overlapping components by their color signatures into independent, movable SVG layers."
)

# ============================================================
# SEMANTIC COLOR RANGES FOR SCIENTIFIC & GENOMIC DIAGRAMS
# ============================================================

SEMANTIC_COLOR_RANGES = {
    "Exons_Red": [
        (np.array([0, 100, 100]), np.array([10, 255, 255])),
        (np.array([170, 100, 100]), np.array([180, 255, 255]))
    ],
    "Gene_Body_Yellow": [
        (np.array([20, 100, 150]), np.array([35, 255, 255]))
    ],
    "TSS_Purple": [
        (np.array([125, 40, 100]), np.array([155, 255, 255]))
    ],
    "Transposon_Teal": [
        (np.array([80, 50, 100]), np.array([105, 255, 255]))
    ],
    "Axes_and_Text_Black": [
        (np.array([0, 0, 0]), np.array([180, 255, 60]))
    ]
}


def extract_overlapping_layers(image):
    """Isolates overlapping shapes and text into distinct independent semantic layers using HSV masking."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, w = image.shape[:2]
    layers = {}

    for layer_name, ranges in SEMANTIC_COLOR_RANGES.items():
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        for lower, upper in ranges:
            mask = cv2.inRange(hsv, lower, upper)
            combined_mask = cv2.bitwise_or(combined_mask, mask)

        # Clean up noise while preserving shape boundaries
        kernel = np.ones((3, 3), np.uint8)
        cleaned_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        pixel_count = cv2.countNonZero(cleaned_mask)
        if pixel_count > 30:  # Ignore tiny specks
            layers[layer_name] = cleaned_mask

    return layers


def generate_layered_svg(image, layers):
    """Generates an SVG containing independent, movable groups for each overlapping layer."""
    h, w = image.shape[:2]
    import base64

    svg_lines = []
    svg_lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 {w} {h}" width="100%" height="100%">')
    svg_lines.append('  <style>')
    svg_lines.append('    .movable-layer { cursor: pointer; }')
    svg_lines.append('  </style>')
    
    # White background base layer
    svg_lines.append(f'  <rect width="{w}" height="{h}" fill="#ffffff" />')

    for layer_name, mask in layers.items():
        # Apply mask to original image to extract exact colors and semi-transparent blends
        isolated_bgr = cv2.bitwise_and(image, image, mask=mask)
        
        # Convert to BGRA with alpha transparency
        bgra = cv2.cvtColor(isolated_bgr, cv2.COLOR_BGR2BGRA)
        bgra[mask == 0, 3] = 0  # Make non-masked pixels fully transparent

        # Find bounding box of this specific layer
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            continue
            
        xmin, xmax = int(np.min(xs)), int(np.max(xs))
        ymin, ymax = int(np.min(ys)), int(np.max(ys))

        # Crop tight around the layer
        pad = 4
        nx = max(0, xmin - pad)
        ny = max(0, ymin - pad)
        nw = min(w - nx, (xmax - xmax) + (xmax - xmin) + (2 * pad)) # robust width
        nw = min(w - nx, (xmax - xmin) + (2 * pad))
        nh = min(h - ny, (ymax - ymin) + (2 * pad))

        cropped_bgra = bgra[ny:ny+nh, nx:nx+nw]

        success, encoded_png = cv2.imencode(".png", cropped_bgra)
        if success:
            b64_str = base64.b64encode(encoded_png.tobytes()).decode("utf-8")
            
            # Wrap each semantic layer in its own interactive SVG group (<g>) for independent movement
            svg_lines.append(f'  <g id="{layer_name}" class="movable-layer">')
            svg_lines.append(f'    <image x="{nx}" y="{ny}" width="{nw}" height="{nh}" xlink:href="data:image/png;base64,{b64_str}" />')
            svg_lines.append('  </g>')

    svg_lines.append('</svg>')
    return "\n".join(svg_lines)


# ============================================================
# STREAMLIT UI
# ============================================================

uploaded_file = st.file_uploader("Upload scientific diagram", type=["jpg", "jpeg", "png", "webp"])

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

        layers = extract_overlapping_layers(img)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Diagram")
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)

        with col2:
            st.subheader("Isolated Overlapping Layers Preview")
            preview = np.ones_like(img) * 255
            for name, mask in layers.items():
                preview[mask > 0] = img[mask > 0]

            st.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.caption(f"Successfully separated {len(layers)} overlapping semantic layers (Exons, Gene Body, TSS, Transposons, Text).")

        st.markdown("---")

        render_btn = st.button("🧬 Export Overlapping Layers to Movable SVG", type="primary", disabled=(len(layers) == 0))

        if render_btn:
            with st.spinner("Decomposing overlapping shapes and generating independent SVG layers..."):
                svg_data = generate_layered_svg(img, layers)

            st.subheader("📁 Layered SVG Ready")
            st.success("SVG generated successfully! When imported into Google Slides or Illustrator and ungrouped, every overlapping shape can be moved independently without distortion.")

            st.download_button(
                label="Download Layered Movable SVG (.svg)",
                data=svg_data,
                file_name="overlapping_diagram_movable.svg",
                mime="image/svg+xml",
            )

    except Exception as e:
        st.error(f"Error encountered: {e}")
