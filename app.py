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
    page_title="Single-File Movable Diagram PPTX Studio",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Single-File Movable Diagram PowerPoint Studio")
st.write(
    "Upload your scientific diagram. The tool generates a **single PowerPoint file** where every "
    "shape is precisely positioned to form your exact diagram, while remaining 100% independently movable "
    "and ready for animations and transitions in Google Slides or PowerPoint."
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


def generate_single_file_powerpoint(img, color_parts_map, color_sentences, color_order):
    """Generates a single PPTX file with transparent image shapes positioned to form the exact diagram."""
    prs = Presentation()
    prs.slide_width = Inches(13.333)  # Widescreen format
    prs.slide_height = Inches(7.5)

    blank_layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(blank_layout)

    # Set solid white background so it doesn't appear transparent/checkered in Google Slides
    background = slide.background
    fill = background.fill
    fill.solid()
    fill.fore_color.rgb = RGBColor(255, 255, 255)

    img_h, img_w = img.shape[:2]

    # Define layout dimensions on the slide (Left side for the diagram)
    canvas_left = Inches(0.8)
    canvas_top = Inches(0.8)
    canvas_max_width = Inches(7.0)
    canvas_max_height = Inches(5.8)

    # Calculate proportional scaling to fit the slide perfectly
    scale = min(canvas_max_width / img_w, canvas_max_height / img_h)

    # 1. Add each detected shape as an individual transparent movable picture object in memory
    for color_name in color_order:
        if color_name not in color_parts_map:
            continue
        
        for part in color_parts_map[color_name]:
            xmin, ymin, xmax, ymax = part["bbox"]
            
            # Add small padding for clean edges
            pad = 2
            xmin_p = max(0, xmin - pad)
            ymin_p = max(0, ymin - pad)
            xmax_p = min(img_w - 1, xmax + pad)
            ymax_p = min(img_h - 1, ymax + pad)

            # Crop region and apply transparency mask
            cropped_bgr = img[ymin_p:ymax_p+1, xmin_p:xmax_p+1]
            cropped_mask = part["mask"][ymin_p:ymax_p+1, xmin_p:xmax_p+1]

            bgra = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2BGRA)
            bgra[~cropped_mask, 3] = 0  # Transparent outside mask

            pil_part = Image.fromarray(cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGBA))
            
            # Save into memory stream
            img_stream = io.BytesIO()
            pil_part.save(img_stream, format="PNG")
            img_stream.seek(0)

            # Calculate precise position matching the original diagram layout
            p_left = canvas_left + Inches(xmin_p * scale)
            p_top = canvas_top + Inches(ymin_p * scale)
            p_width = Inches((xmax_p - xmin_p) * scale)
            p_height = Inches((ymax_p - ymin_p) * scale)

            # Insert into slide as an independent movable object
            slide.shapes.add_picture(img_stream, p_left, p_top, width=p_width, height=p_height)

    # 2. Add movable text boxes on the right side for captions and animation notes
    text_left = Inches(8.3)
    text_top = Inches(0.8)
    text_width = Inches(4.5)

    for color_name in color_order:
        if color_name not in color_parts_map:
            continue
        
        sentence = color_sentences.get(color_name, f"Details for {color_name}")

        box = slide.shapes.add_textbox(text_left, text_top, text_width, Inches(1.1))
        tf = box.text_frame
        tf.word_wrap = True

        p_head = tf.paragraphs[0]
        p_head.text = f"■ {color_name} Component"
        p_head.font.size = Pt(13)
        p_head.font.bold = True
        p_head.font.color.rgb = RGBColor(0, 102, 204)

        p_desc = tf.add_paragraph()
        p_desc.text = sentence
        p_desc.font.size = Pt(11)
        p_desc.font.color.rgb = RGBColor(50, 50, 50)

        text_top += Inches(1.2)

    # Save presentation to memory buffer
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

        st.sidebar.header("⚙️ Component Settings")
        color_sentences = {}

        if color_parts_map:
            st.sidebar.write("Customize descriptions per shape category:")
            color_order = list(color_parts_map.keys())

            for color_name, parts in color_parts_map.items():
                with st.sidebar.expander(f"Color: {color_name} ({len(parts)} objects)"):
                    sentence_input = st.text_input(
                        "Animation / Description Note",
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
            st.subheader("Isolated Moveable Objects Preview")
            preview = img.copy()
            total_shapes = 0

            for color_name, parts in color_parts_map.items():
                total_shapes += len(parts)
                for p in parts:
                    xmin, ymin, xmax, ymax = p["bbox"]
                    cv2.rectangle(preview, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)

            st.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.caption(f"Mapped {total_shapes} independent movable objects.")

        st.markdown("---")

        render_btn = st.button("📊 Export Movable Diagram Presentation (.pptx)", type="primary", disabled=(len(color_parts_map) == 0))

        if render_btn:
            with st.spinner("Building single-file PowerPoint presentation..."):
                ppt_bytes = generate_single_file_powerpoint(img, color_parts_map, color_sentences, color_order)

            st.subheader("📁 Presentation Ready")
            st.success("PowerPoint generated successfully! Upload this `.pptx` file into Google Slides to view and animate your diagram.")

            st.download_button(
                label="Download PowerPoint Presentation (.pptx)",
                data=ppt_bytes,
                file_name="movable_diagram_presentation.pptx",
                mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
            )

    except Exception as e:
        st.error(f"Error encountered: {e}")
