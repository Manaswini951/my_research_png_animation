import io
import math
import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor

st.set_page_config(
    page_title="Scientific Diagram to PowerPoint Converter",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Scientific Diagram to PowerPoint Converter")
st.write(
    "Upload a scientific diagram or illustration. The tool recognizes shapes based on colors "
    "and exports them into an editable PowerPoint presentation format (.pptx) along with their captions."
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

        if coverage > 0.08:  # Ignore tiny speckles
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

            base_y = float(np.max(ys))
            base_x = float(np.mean(xs[ys == int(base_y)]))

            tip_y = float(np.min(ys))
            tip_x = float(np.mean(xs[ys == int(tip_y)]))

            length = max(10.0, float(math.hypot(tip_x - base_x, tip_y - base_y)))
            expanded = cv2.dilate(comp, np.ones((5, 5), np.uint8), iterations=1)

            parts.append({
                "mask": expanded > 0,
                "base": (base_x, base_y),
                "tip": (tip_x, tip_y),
                "center": (float(np.mean(xs)), float(np.mean(ys))),
                "length": length,
                "area": area,
            })

        if parts:
            color_parts_map[color_name] = parts

    return color_parts_map


def generate_powerpoint(img, color_parts_map, color_sentences, color_order):
    """Generates a PowerPoint presentation containing the diagram and separated component details."""
    prs = Presentation()
    
    # Slide 1: Title & Full Diagram Overview with all captions
    blank_slide_layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(blank_slide_layout)

    # Save original image temporarily to insert into PPT
    temp_img_path = "temp_diagram.png"
    cv2.imwrite(temp_img_path, img)

    # Add Diagram Image on Left Side
    slide.shapes.add_picture(temp_img_path, Inches(0.8), Inches(1.0), width=Inches(5.0))

    # Add Title
    title_box = slide.shapes.add_textbox(Inches(6.2), Inches(0.8), Inches(6.5), Inches(0.8))
    tf = title_box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Scientific Diagram Analysis"
    p.font.size = Pt(24)
    p.font.bold = True
    p.font.color.rgb = RGBColor(30, 30, 30)

    # Add Component Sentences / Descriptions as Editable Text Boxes on Right Side
    top_offset = Inches(1.8)
    for color_name in color_order:
        if color_name not in color_parts_map:
            continue
        sentence = color_sentences.get(color_name, f"Component: {color_name}")
        
        box = slide.shapes.add_textbox(Inches(6.2), top_offset, Inches(6.5), Inches(0.9))
        box_tf = box.text_frame
        box_tf.word_wrap = True
        
        p_head = box_tf.paragraphs[0]
        p_head.text = f"• [{color_name}]"
        p_head.font.size = Pt(14)
        p_head.font.bold = True
        p_head.font.color.rgb = RGBColor(0, 102, 204)

        p_desc = box_tf.add_paragraph()
        p_desc.text = sentence
        p_desc.font.size = Pt(13)
        p_desc.font.color.rgb = RGBColor(50, 50, 50)

        top_offset += Inches(1.0)

    # Slide 2+: Individual breakdown per detected color region
    for color_name in color_order:
        if color_name not in color_parts_map:
            continue
        
        detail_slide = prs.slides.add_slide(blank_slide_layout)
        
        # Section Title
        d_title_box = detail_slide.shapes.add_textbox(Inches(0.8), Inches(0.6), Inches(11.5), Inches(0.8))
        dtf = d_title_box.text_frame
        dp = dtf.paragraphs[0]
        dp.text = f"Component Focus: {color_name}"
        dp.font.size = Pt(22)
        dp.font.bold = True
        dp.font.color.rgb = RGBColor(0, 102, 204)

        # Editable text area for detailed notes / sentences
        note_box = detail_slide.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(11.5), Inches(1.5))
        ntf = note_box.text_frame
        ntf.word_wrap = True
        np_main = ntf.paragraphs[0]
        np_main.text = color_sentences.get(color_name, "")
        np_main.font.size = Pt(16)
        np_main.font.color.rgb = RGBColor(40, 40, 40)

        # Add image reminder on slide
        detail_slide.shapes.add_picture(temp_img_path, Inches(0.8), Inches(3.4), width=Inches(4.5))

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
            scale = 1000.0 / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        detected_colors = extract_accurate_colors(img)
        color_parts_map = detect_parts_per_color(detected_colors)

        st.sidebar.header("⚙️ PowerPoint Configuration Dashboard")
        color_sentences = {}

        if color_parts_map:
            st.sidebar.write("Customize sentences / labels per detected color region:")
            color_order = list(color_parts_map.keys())

            for color_name, parts in color_parts_map.items():
                with st.sidebar.expander(f"Color: {color_name} ({len(parts)} regions)"):
                    sentence_input = st.text_input(
                        "Caption / Explanation Sentence",
                        value=f"Description for component: {color_name}",
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
            st.subheader("Detected Scientific Color Regions")
            preview = img.copy()

            for color_name, parts in color_parts_map.items():
                for p in parts:
                    cv2.circle(preview, (int(p["base"][0]), int(p["base"][1])), 6, (0, 255, 0), -1)
                    cv2.circle(preview, (int(p["tip"][0]), int(p["tip"][1])), 4, (0, 0, 255), -1)
                    cv2.line(preview, (int(p["base"][0]), int(p["base"][1])), (int(p["tip"][0]), int(p["tip"][1])), (255, 255, 0), 2)

            st.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB), use_container_width=True)

        st.markdown("---")

        render_btn = st.button("📊 Export to PowerPoint (.pptx)", type="primary", disabled=(len(color_parts_map) == 0))

        if render_btn:
            with st.spinner("Generating PowerPoint presentation with editable shapes and captions..."):
                ppt_bytes = generate_powerpoint(img, color_parts_map, color_sentences, color_order)

            st.subheader("📁 PowerPoint Export Ready")
            st.success("Presentation generated successfully! You can now download and freely edit text and elements.")

            st.download_button(
                label="Download PowerPoint Presentation (.pptx)",
                data=ppt_bytes,
                file_name="scientific_diagram_presentation.pptx",
                mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
            )

    except Exception as e:
        st.error(f"Error encountered: {e}")
