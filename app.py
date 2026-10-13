import io
import math
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor

st.set_page_config(
    page_title="Scientific Diagram to Movable PPTX Objects",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Scientific Diagram to Movable PowerPoint Objects")
st.write(
    "Upload a scientific diagram. The tool breaks down every detected shape into an independent, "
    "movable object in PowerPoint, along with movable text boxes for your captions."
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
        min_area = 40

        for i in range(1, num_labels):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < min_area:
                continue

            comp = (labels == i).astype(np.uint8) * 255
            ys, xs = np.where(comp > 0)
            if len(xs) < 10:
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


def generate_movable_objects_powerpoint(img, color_parts_map, color_sentences, color_order):
    """Generates a PPTX where every shape and text box is an independent, movable object."""
    prs = Presentation()
    prs.slide_width = Inches(13.333)  # Widescreen format
    prs.slide_height = Inches(7.5)

    blank_layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(blank_layout)

    img_h, img_w = img.shape[:2]

    # Canvas mapping dimensions on PowerPoint (Left side area for shapes)
    canvas_left = Inches(0.8)
    canvas_top = Inches(1.0)
    canvas_max_width = Inches(7.0)
    canvas_max_height = Inches(6.0)

    # Calculate scaling factor to fit diagram nicely while keeping relative positions
    scale = min(canvas_max_width / img_w, canvas_max_height / img_h)

    # 1. Add each detected shape as an individual transparent movable picture object
    part_counter = 0
    for color_name in color_order:
        if color_name not in color_parts_map:
            continue
        
        for idx, part in enumerate(color_parts_map[color_name]):
            xmin, ymin, xmax, ymax = part["bbox"]
            
            # Add padding
            pad = 4
            xmin = max(0, xmin - pad)
            ymin = max(0, ymin - pad)
            xmax = min(img_w - 1, xmax + pad)
            ymax = min(img_h - 1, ymax + pad)

            # Crop the bounding box region
            cropped_bgr = img[ymin:ymax+1, xmin:xmax+1]
            cropped_mask = part["mask"][ymin:ymax+1, xmin:xmax+1]

            # Create RGBA image with transparent background outside the shape mask
            bgra = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2BGRA)
            bgra[~cropped_mask, 3] = 0  # Set alpha to 0 (transparent)

            # Save temporary transparent part image
            part_img_path = f"temp_shape_{color_name}_{idx}.png"
            pil_part = Image.fromarray(cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGBA))
            pil_part.save(part_img_path)

            # Compute PowerPoint coordinates for this movable shape
            p_left = canvas_left + Inches(xmin * scale)
            p_top = canvas_top + Inches(ymin * scale)
            p_width = Inches((xmax - xmin) * scale)
            p_height = Inches((ymax - ymin) * scale)

            # Add as independent picture object
            slide.shapes.add_picture(part_img_path, p_left, p_top, width=p_width, height=p_height)
            part_counter += 1

    # 2. Add movable text boxes on the right side for each color category
    text_left = Inches(8.5)
    text_top = Inches(1.0)
    text_width = Inches(4.3)

    for color_name in color_order:
        if color_name not in color_parts_map:
            continue
        
        sentence = color_sentences.get(color_name, f"Component: {color_name}")

        # Create an independent movable text box
        box = slide.shapes.add_textbox(text_left, text_top, text_width, Inches(1.2))
        tf = box.text_frame
        tf.word_wrap = True

        p_head = tf.paragraphs[0]
        p_head.text = f"■ {color_name} Region"
        p_head.font.size = Pt(14)
        p_head.font.bold = True
        p_head.font.color.rgb = RGBColor(0, 102, 204)

        p_desc = tf.add_paragraph()
        p_desc.text = sentence
        p_desc.font.size = Pt(13)
        p_desc.font.color.rgb = RGBColor(40, 40, 40)

        text_top += Inches(1.3)

    # Save presentation to buffer
    ppt_io = io.BytesIO()
    prs.save(ppt_io)
    ppt_io.seek(0)
    return ppt_io.getvalue()


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

        st.sidebar.header("⚙️ Movable Objects Configuration")
        color_sentences = {}

        if color_parts_map:
            st.sidebar.write("Customize movable text annotations per color region:")
            color_order = list(color_parts_map.keys())

            for color_name, parts in color_parts_map.items():
                with st.sidebar.expander(f"Color: {color_name} ({len(parts)} shapes)"):
                    sentence_input = st.text_input(
                        "Caption / Explanation",
                        value=f"Detailed notes for {color_name} component.",
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
            st.subheader("Detected Movable Shapes")
            preview = img.copy()
            total_shapes = 0

            for color_name, parts in color_parts_map.items():
                total_shapes += len(parts)
                for p in parts:
                    xmin, ymin, xmax, ymax = p["bbox"]
                    cv2.rectangle(preview, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

            st.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.caption(f"Total individual shapes isolated: {total_shapes}")

        st.markdown("---")

        render_btn = st.button("📊 Export Movable Objects to PowerPoint (.pptx)", type="primary", disabled=(len(color_parts_map) == 0))

        if render_btn:
            with st.spinner("Isolating shapes, generating transparent layers, and building PowerPoint..."):
                ppt_bytes = generate_movable_objects_powerpoint(img, color_parts_map, color_sentences, color_order)

            st.subheader("📁 Presentation Ready")
            st.success("PowerPoint generated! Every shape and text box is now a separate, fully movable object.")

            st.download_button(
                label="Download PowerPoint Presentation (.pptx)",
                data=ppt_bytes,
                file_name="movable_scientific_shapes.pptx",
                mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
            )

    except Exception as e:
        st.error(f"Error encountered: {e}")
