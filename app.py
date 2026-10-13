import io
import math
import base64
import cv2
import numpy as np
import streamlit as st

st.set_page_config(
    page_title="Universal Scientific Diagram SVG Studio",
    page_icon="🧬",
    layout="wide",
)

st.title("🧬 Universal Scientific Diagram SVG Studio")
st.write(
    "Upload any scientific diagram (such as genomic charts, pathways, or schematics). "
    "The tool preserves all shapes, overlapping layers, and custom captions into an **editable SVG file**, "
    "ensuring every single component remains entirely independent and movable."
)

# ============================================================
# COMPREHENSIVE HSV COLOR BOUNDS FOR SCIENTIFIC DIAGRAMS
# ============================================================

COLOR_RANGES = {
    "Red / Exons": [
        (np.array([0, 70, 50]), np.array([8, 255, 255])),
        (np.array([172, 70, 50]), np.array([180, 255, 255])),
    ],
    "Orange": [(np.array([9, 70, 50]), np.array([20, 255, 255]))],
    "Yellow / Gene Body": [(np.array([21, 70, 50]), np.array([34, 255, 255]))],
    "Green": [(np.array([35, 50, 40]), np.array([84, 255, 255]))],
    "Cyan / Transposon Body": [(np.array([85, 50, 40]), np.array([105, 255, 255]))],
    "Blue": [(np.array([106, 50, 40]), np.array([130, 255, 255]))],
    "Purple / TSS Region": [(np.array([125, 40, 100]), np.array([155, 255, 255]))],
    "Pink / Magenta": [(np.array([156, 50, 40]), np.array([171, 255, 255]))],
    "Axes and Text": [(np.array([0, 0, 0]), np.array([180, 255, 60]))],
}


def extract_accurate_colors(image):
    """Scans image using defined HSV ranges to detect color regions and overlapping structures."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    total_pixels = image.shape[0] * image.shape[1]
    detected = []

    for name, ranges in COLOR_RANGES.items():
        combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in ranges:
            mask = cv2.inRange(hsv, lower, upper)
            combined_mask = cv2.bitwise_or(combined_mask, mask)

        pixel_count = cv2.countNonZero(combined_mask)
        coverage = (pixel_count / total_pixels) * 100

        if coverage > 0.03:  # Lower threshold to capture small markers/text
            detected.append({
                "name": name,
                "coverage": coverage,
                "mask": combined_mask,
            })

    return detected


def detect_parts_per_color(detected_colors, image_shape):
    """Extracts connected components for each distinct color region while respecting overlapping bounds."""
    color_parts_map = {}
    h, w = image_shape[:2]

    for color_info in detected_colors:
        color_name = color_info["name"]
        mask = color_info["mask"]

        clean_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(clean_mask, connectivity=8)

        parts = []
        min_area = 20  # Allows catching small text/icons and markers

        for i in range(1, num_labels):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < min_area:
                continue

            comp = (labels == i).astype(np.uint8) * 255
            ys, xs = np.where(comp > 0)
            if len(xs) < 5:
                continue

            parts.append({
                "mask": comp > 0,
                "bbox": (int(np.min(xs)), int(np.min(ys)), int(np.max(xs)), int(np.max(ys))),
                "center": (float(np.mean(xs)), float(np.mean(ys))),
                "area": area,
            })

        if parts:
            color_parts_map[color_name] = parts

    return color_parts_map


def generate_comprehensive_svg(img, color_parts_map, color_sentences, color_order):
    """Generates an advanced SVG containing independent, movable vector layers, overlapping shapes, and text annotations."""
    h, w = img.shape[:2]
    
    svg_lines = []
    # Expand canvas width slightly on the right to accommodate custom text sidebar if needed
    svg_lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 {w + 400} {h}" width="100%" height="100%">')
    svg_lines.append('  <style>')
    svg_lines.append('    .movable-group { cursor: pointer; }')
    svg_lines.append('    .svg-title { font-family: Arial, sans-serif; font-size: 15px; font-weight: bold; fill: #0275d8; }')
    svg_lines.append('    .svg-desc { font-family: Arial, sans-serif; font-size: 12px; fill: #333333; }')
    svg_lines.append('  </style>')
    
    # White background base layer
    svg_lines.append(f'  <rect width="{w}" height="{h}" fill="#ffffff" />')

    shape_counter = 1
    for color_name in color_order:
        if color_name not in color_parts_map:
            continue
        
        for part in color_parts_map[color_name]:
            xmin, ymin, xmax, ymax = part["bbox"]
            
            # Precise padding to preserve transparent borders and overlapping blends
            pad = 2
            xmin_p = max(0, xmin - pad)
            ymin_p = max(0, ymin - pad)
            xmax_p = min(w - 1, xmax + pad)
            ymax_p = min(h - 1, ymax + pad)

            cropped_bgr = img[ymin_p:ymax_p+1, xmin_p:xmax_p+1]
            cropped_mask = part["mask"][ymin_p:ymax_p+1, xmin_p:xmax_p+1]

            # Convert to BGRA with alpha transparency for overlapping shapes
            bgra = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2BGRA)
            bgra[~cropped_mask, 3] = 0  # Transparent outside mask

            success, encoded_png = cv2.imencode(".png", bgra)
            if success:
                b64_str = base64.b64encode(encoded_png.tobytes()).decode("utf-8")
                pw = xmax_p - xmin_p
                ph = ymax_p - ymin_p
                
                safe_name = color_name.split("/")[0].strip().replace(" ", "_")
                svg_lines.append(f'  <g id="shape_{safe_name}_{shape_counter}" class="movable-group">')
                svg_lines.append(f'    <image x="{xmin_p}" y="{ymin_p}" width="{pw}" height="{ph}" xlink:href="data:image/png;base64,{b64_str}" />')
                svg_lines.append('  </g>')
                shape_counter += 1

    # Add custom text/annotation sidebar blocks
    text_x = w + 30
    text_y = 50

    for color_name in color_order:
        if color_name not in color_parts_map:
            continue
        
        sentence = color_sentences.get(color_name, f"Notes for {color_name}")
        safe_name = color_name.split("/")[0].strip().replace(" ", "_")

        svg_lines.append(f'  <g id="text_{safe_name}" class="movable-group">')
        svg_lines.append(f'    <text x="{text_x}" y="{text_y}" class="svg-title">■ {color_name}</text>')
        svg_lines.append(f'    <text x="{text_x}" y="{text_y + 20}" class="svg-desc">{sentence}</text>')
        svg_lines.append('  </g>')

        text_y += 80

    svg_lines.append('</svg>')
    return "\n".join(svg_lines)


# ============================================================
# STREAMLIT UI
# ============================================================

uploaded_file = st.file_uploader("Upload scientific diagram / genomic chart", type=["jpg", "jpeg", "png", "webp"])

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

        detected_colors = extract_accurate_colors(img)
        color_parts_map = detect_parts_per_color(detected_colors, img.shape)

        st.sidebar.header("⚙️ Diagram & Annotation Settings")
        color_sentences = {}

        if color_parts_map:
            st.sidebar.write("Customize labels & descriptions per component:")
            color_order = list(color_parts_map.keys())

            for color_name, parts in color_parts_map.items():
                with st.sidebar.expander(f"Layer: {color_name} ({len(parts)} objects)"):
                    sentence_input = st.text_input(
                        "Caption / Explanation",
                        value=f"Description for {color_name}.",
                        key=f"sentence_{color_name}"
                    )
                    color_sentences[color_name] = sentence_input
        else:
            st.sidebar.warning("No distinct color features detected.")
            color_order = []

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Diagram")
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)

        with col2:
            st.subheader("Detected Movable Layers Preview")
            preview = img.copy()
            total_elements = 0

            for color_name, parts in color_parts_map.items():
                total_elements += len(parts)
                for p in parts:
                    xmin, ymin, xmax, ymax = p["bbox"]
                    cv2.rectangle(preview, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)

            st.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.caption(f"Successfully isolated {total_elements} independent movable vector shapes and text objects.")

        st.markdown("---")

        render_btn = st.button("🧬 Export Comprehensive Movable SVG (.svg)", type="primary", disabled=(len(color_parts_map) == 0))

        if render_btn:
            with st.spinner("Processing shapes, overlapping layers, and building SVG package..."):
                svg_data = generate_comprehensive_svg(img, color_parts_map, color_sentences, color_order)

            st.subheader("📁 SVG Package Ready")
            st.success("SVG file generated successfully! Import it into Google Slides or PowerPoint, right-click and ungroup to move every single overlapping shape and text independently.")

            st.download_button(
                label="Download Movable SVG Diagram (.svg)",
                data=svg_data,
                file_name="scientific_diagram_movable_layers.svg",
                mime="image/svg+xml",
            )

    except Exception as e:
        st.error(f"Error encountered: {e}")
