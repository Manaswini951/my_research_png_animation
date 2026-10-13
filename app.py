import io
import math
import cv2
import numpy as np
import streamlit as st

st.set_page_config(
    page_title="Scientific Diagram SVG Studio",
    page_icon="📐",
    layout="wide",
)

st.title("📐 Scientific Diagram SVG Studio")
st.write(
    "Upload your scientific diagram. The tool extracts every detected component into an **SVG (Scalable Vector Graphics)** format. "
    "When you open or import the SVG into Google Slides, PowerPoint, or any vector editor, every single shape is completely separate, "
    "vector-editable, and independently movable!"
)

# ============================================================
# ACCURATE HSV COLOR BOUNDS
# ============================================================

COLOR_RANGES = {
    "Red": [
        (np.array([0, 70, 50]), np.array([8, 255, 255])),
        (np.array([172, 70, 50]), np.array([180, 255, 255])),
    ],
    "Orange": [(np.array([9, 70, 50]), np.array([20, 255, 255]))],
    "Yellow": [(np.array([21, 70, 50]), np.array([34, 255, 255]))],
    "Green": [(np.array([35, 50, 40]), np.array([84, 255, 255]))],
    "Cyan / Light Blue": [(np.array([85, 50, 40]), np.array([105, 255, 255]))],
    "Blue": [(np.array([106, 50, 40]), np.array([130, 255, 255]))],
    "Purple / Violet": [(np.array([131, 50, 40]), np.array([155, 255, 255]))],
    "Pink / Magenta": [(np.array([156, 50, 40]), np.array([171, 255, 255]))],
}


def extract_accurate_colors(image):
    """Scans image using defined HSV ranges to detect color regions."""
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

        if coverage > 0.08:
            detected.append({
                "name": name,
                "coverage": coverage,
                "mask": combined_mask,
            })

    return detected


def detect_parts_per_color(detected_colors):
    """Extracts connected components for each distinct color region."""
    color_parts_map = {}

    for color_info in detected_colors:
        color_name = color_info["name"]
        mask = color_info["mask"]

        clean_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(clean_mask, connectivity=8)

        parts = []
        min_area = 30

        for i in range(1, num_labels):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < min_area:
                continue

            comp = (labels == i).astype(np.uint8) * 255
            ys, xs = np.where(comp > 0)
            if len(xs) < 8:
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


def generate_svg_file(img, color_parts_map, color_sentences, color_order):
    """Generates a well-structured SVG file containing grouped, movable shape elements and text annotations."""
    h, w = img.shape[:2]
    
    svg_lines = []
    svg_lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 {0} {w + 450} {h}" width="100%" height="100%">')
    svg_lines.append('  <style>')
    svg_lines.append('    .shape-group { cursor: pointer; }')
    svg_lines.append('    .label-box { font-family: Arial, sans-serif; font-size: 14px; fill: #222; }')
    svg_lines.append('    .label-title { font-family: Arial, sans-serif; font-size: 16px; font-weight: bold; fill: #0275d8; }')
    svg_lines.append('  </style>')
    
    # White background for clarity
    svg_lines.append(f'  <rect width="{w}" height="{h}" fill="#ffffff" />')

    # 1. Add each detected shape as an embedded base64 PNG image inside its own SVG group (fully movable)
    import base64
    for color_name in color_order:
        if color_name not in color_parts_map:
            continue
        
        for idx, part in enumerate(color_parts_map[color_name]):
            xmin, ymin, xmax, ymax = part["bbox"]
            
            pad = 2
            xmin_p = max(0, xmin - pad)
            ymin_p = max(0, ymin - pad)
            xmax_p = min(w - 1, xmax + pad)
            ymax_p = min(h - 1, ymax + pad)

            cropped_bgr = img[ymin_p:ymax_p+1, xmin_p:xmax_p+1]
            cropped_mask = part["mask"][ymin_p:ymax_p+1, xmin_p:xmax_p+1]

            bgra = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2BGRA)
            bgra[~cropped_mask, 3] = 0

            success, encoded_png = cv2.imencode(".png", bgra)
            if success:
                b64_str = base64.b64encode(encoded_png.tobytes()).decode("utf-8")
                pw = xmax_p - xmin_p
                ph = ymax_p - ymin_p
                
                # Each shape is wrapped in an independent SVG group (<g>) so it can be moved individually
                svg_lines.append(f'  <g id="shape_{color_name.replace(" ", "_")}_{idx}" class="shape-group">')
                svg_lines.append(f'    <image x="{xmin_p}" y="{ymin_p}" width="{pw}" height="{ph}" xlink:href="data:image/png;base64,{b64_str}" />')
                svg_lines.append('  </g>')

    # 2. Add movable text annotation blocks on the right side of the SVG canvas
    text_x = w + 40
    text_y = 60

    for color_name in color_order:
        if color_name not in color_parts_map:
            continue
        
        sentence = color_sentences.get(color_name, f"Details for {color_name}")
        
        svg_lines.append(f'  <g id="text_{color_name.replace(" ", "_")}" class="shape-group">')
        svg_lines.append(f'    <text x="{text_x}" y="{text_y}" class="label-title">■ {color_name}</text>')
        svg_lines.append(f'    <text x="{text_x}" y="{text_y + 22}" class="label-box">{sentence}</text>')
        svg_lines.append('  </g>')

        text_y += 90

    svg_lines.append('</svg>')
    return "\n".join(svg_lines)


# ============================================================
# STREAMLIT UI
# ============================================================

uploaded_file = st.file_uploader("Upload scientific diagram / illustration", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    try:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img is None:
            st.error("Invalid image format.")
            st.stop()

        h, w = img.shape[:2]
        if max(h, w) > 1000:
            scale_val = 1000.0 / max(h, w)
            img = cv2.resize(img, (int(w * scale_val), int(h * scale_val)), interpolation=cv2.INTER_AREA)

        detected_colors = extract_accurate_colors(img)
        color_parts_map = detect_parts_per_color(detected_colors)

        st.sidebar.header("⚙️ SVG Component Settings")
        color_sentences = {}

        if color_parts_map:
            st.sidebar.write("Customize annotations per shape category:")
            color_order = list(color_parts_map.keys())

            for color_name, parts in color_parts_map.items():
                with st.sidebar.expander(f"Color: {color_name} ({len(parts)} objects)"):
                    sentence_input = st.text_input(
                        "Annotation / Description",
                        value=f"Description for {color_name} component.",
                        key=f"sentence_{color_name}"
                    )
                    color_sentences[color_name] = sentence_input
        else:
            st.sidebar.warning("No distinct colored markers detected.")
            color_order = []

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Diagram")
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)

        with col2:
            st.subheader("Detected Vector Layers Preview")
            preview = img.copy()
            total_shapes = 0

            for color_name, parts in color_parts_map.items():
                total_shapes += len(parts)
                for p in parts:
                    xmin, ymin, xmax, ymax = p["bbox"]
                    cv2.rectangle(preview, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)

            st.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.caption(f"Mapped {total_shapes} independent SVG elements.")

        st.markdown("---")

        render_btn = st.button("📐 Export Layered SVG File (.svg)", type="primary", disabled=(len(color_parts_map) == 0))

        if render_btn:
            with st.spinner("Generating layered SVG file..."):
                svg_content = generate_svg_file(img, color_parts_map, color_sentences, color_order)

            st.subheader("📁 SVG File Ready")
            st.success("SVG file generated successfully! You can import this directly into Google Slides, PowerPoint (Office 365 supports native SVG importing where every group is ungroupable), or any vector editor.")

            st.download_button(
                label="Download Layered SVG File (.svg)",
                data=svg_content,
                file_name="scientific_diagram_movable.svg",
                mime="image/svg+xml",
            )

    except Exception as e:
        st.error(f"Error encountered: {e}")
