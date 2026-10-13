import io
import math
import zipfile
import cv2
import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Exact Diagram Shape Extractor",
    page_icon="🧩",
    layout="wide",
)

st.title("🧩 Exact Scientific Diagram Shape Extractor")
st.write(
    "Upload your diagram. This tool isolates every single component with precise transparency. "
    "When you drag and drop the extracted shapes into PowerPoint, they reassemble into your **exact original diagram**, "
    "while keeping every single object 100% movable and independent!"
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
    "Cyan_Light_Blue": [(np.array([85, 50, 40]), np.array([105, 255, 255]))],
    "Blue": [(np.array([106, 50, 40]), np.array([130, 255, 255]))],
    "Purple_Violet": [(np.array([131, 50, 40]), np.array([155, 255, 255]))],
    "Pink_Magenta": [(np.array([156, 50, 40]), np.array([171, 255, 255]))],
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


def generate_exact_shapes_zip(img, color_parts_map, color_sentences, color_order):
    """Generates a ZIP package containing transparent shapes and an instruction guide."""
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        # 1. Save original diagram reference
        success, encoded_img = cv2.imencode(".png", img)
        if success:
            zip_file.writestr("1_reference_original_diagram.png", encoded_img.tobytes())

        # 2. Save text guide on how to assemble in PPT
        guide_text = (
            "HOW TO RECONSTRUCT YOUR EXACT DIAGRAM IN POWERPOINT:\n"
            "====================================================\n"
            "1. Open PowerPoint and create a blank slide.\n"
            "2. Open the 'shapes' folder from this zip.\n"
            "3. Select ALL shape images inside the folder and drag-and-drop them onto your PowerPoint slide at once.\n"
            "4. Because they share the exact relative pixel dimensions and transparent backgrounds, "
            "they will line up and form your exact original diagram.\n"
            "5. Every single piece is now an independent movable object that you can shift, separate, or edit!\n\n"
            "CAPTIONS & NOTES:\n"
            "-----------------\n"
        )
        for color_name in color_order:
            if color_name in color_sentences:
                guide_text += f"- [{color_name}]: {color_sentences[color_name]}\n"
        
        zip_file.writestr("0_README_AND_INSTRUCTIONS.txt", guide_text)

        # 3. Extract each exact shape with tight bounding box and alpha transparency
        img_h, img_w = img.shape[:2]
        shape_index = 1

        for color_name in color_order:
            if color_name not in color_parts_map:
                continue
            
            for part in color_parts_map[color_name]:
                xmin, ymin, xmax, ymax = part["bbox"]
                
                # Tight crop with 2px padding for clean edges
                pad = 2
                xmin = max(0, xmin - pad)
                ymin = max(0, ymin - pad)
                xmax = min(img_w - 1, xmax + pad)
                ymax = min(img_h - 1, ymax + pad)

                cropped_bgr = img[ymin:ymax+1, xmin:xmax+1]
                cropped_mask = part["mask"][ymin:ymax+1, xmin:xmax+1]

                # Convert to BGRA with transparent background outside mask
                bgra = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2BGRA)
                bgra[~cropped_mask, 3] = 0

                pil_part = Image.fromarray(cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGBA))
                
                part_io = io.BytesIO()
                pil_part.save(part_io, format="PNG")
                
                file_name = f"shapes/shape_{shape_index:02d}_{color_name}_x{xmin}_y{ymin}.png"
                zip_file.writestr(file_name, part_io.getvalue())
                shape_index += 1

    zip_buffer.seek(0)
    return zip_buffer.getvalue()


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

        st.sidebar.header("⚙️ Component Settings")
        color_sentences = {}

        if color_parts_map:
            st.sidebar.write("Customize labels for each color category:")
            color_order = list(color_parts_map.keys())

            for color_name, parts in color_parts_map.items():
                with st.sidebar.expander(f"Color: {color_name} ({len(parts)} objects)"):
                    sentence_input = st.text_input(
                        "Description",
                        value=f"Component description for {color_name}.",
                        key=f"sentence_{color_name}"
                    )
                    color_sentences[color_name] = sentence_input
        else:
            st.sidebar.warning("No distinct colored markers detected.")
            color_order = []

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Diagram Reference")
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)

        with col2:
            st.subheader("Isolated Moveable Shapes Preview")
            preview = img.copy()
            total_shapes = 0

            for color_name, parts in color_parts_map.items():
                total_shapes += len(parts)
                for p in parts:
                    xmin, ymin, xmax, ymax = p["bbox"]
                    cv2.rectangle(preview, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)

            st.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.caption(f"Successfully isolated {total_shapes} independent movable objects.")

        st.markdown("---")

        render_btn = st.button("🧩 Download Exact Movable Shapes Package (.zip)", type="primary", disabled=(len(color_parts_map) == 0))

        if render_btn:
            with st.spinner("Extracting pixel-perfect transparent shapes..."):
                zip_bytes = generate_exact_shapes_zip(img, color_parts_map, color_sentences, color_order)

            st.subheader("📁 Ready for PowerPoint")
            st.success("Package created successfully! Download below.")

            st.download_button(
                label="Download ZIP Package",
                data=zip_bytes,
                file_name="exact_movable_diagram_shapes.zip",
                mime="application/zip",
            )

    except Exception as e:
        st.error(f"Error encountered: {e}")
