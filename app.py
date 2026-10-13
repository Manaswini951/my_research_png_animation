import io
import math
import cv2
import numpy as np
import streamlit as st
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE

st.set_page_config(
    page_title="Native PPTX Shape & Transition Studio",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Native PowerPoint Shape & Transition Studio")
st.write(
    "Upload your diagram. The tool converts detected shapes and texts into **native PowerPoint shapes and text boxes**, "
    "allowing you to freely move them and apply individual PowerPoint transitions and animations to each object!"
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
                "bbox": (int(np.min(xs)), int(np.min(ys)), int(np.max(xs)), int(np.max(ys))),
                "center": (float(np.mean(xs)), float(np.mean(ys))),
                "area": area,
            })

        if parts:
            color_parts_map[color_name] = parts

    return color_parts_map


def generate_native_editable_powerpoint(img, color_parts_map, color_sentences, color_order):
    """Generates a PPTX using native PowerPoint shapes and text boxes so they are fully editable and animatable."""
    prs = Presentation()
    prs.slide_width = Inches(13.333)  # Widescreen format
    prs.slide_height = Inches(7.5)

    blank_layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(blank_layout)

    img_h, img_w = img.shape[:2]

    # Canvas dimensions on PowerPoint for the diagram mapping
    canvas_left = Inches(1.0)
    canvas_top = Inches(1.0)
    canvas_width = Inches(6.5)
    canvas_height = Inches(5.5)

    scale_x = canvas_width / img_w
    scale_y = canvas_height / img_h

    # 1. Add native PowerPoint shapes for each detected component
    for color_name in color_order:
        if color_name not in color_parts_map:
            continue

        for part in color_parts_map[color_name]:
            xmin, ymin, xmax, ymax = part["bbox"]
            
            p_left = canvas_left + Inches(xmin * scale_x)
            p_top = canvas_top + Inches(ymin * scale_y)
            p_w = max(Inches(0.2), Inches((xmax - xmin) * scale_x))
            p_h = max(Inches(0.2), Inches((ymax - ymin) * scale_y))

            # Create a native PowerPoint shape (Rectangle placeholder representing the component)
            shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, p_left, p_top, p_w, p_h)
            
            # Style the shape cleanly so it looks professional and is fully selectable/animatable
            shape.text = color_name.split("/")[0].strip()
            shape.text_frame.word_wrap = True
            
            # Format text inside shape
            for paragraph in shape.text_frame.paragraphs:
                paragraph.font.size = Pt(10)
                paragraph.font.bold = True
                paragraph.font.color.rgb = RGBColor(255, 255, 255)

            # Style shape fill based on category
            fill = shape.fill
            fill.solid()
            
            # Assign distinctive nice colors
            if "Red" in color_name:
                fill.fore_color.rgb = RGBColor(217, 83, 79)
            elif "Green" in color_name:
                fill.fore_color.rgb = RGBColor(92, 184, 92)
            elif "Blue" in color_name or "Cyan" in color_name:
                fill.fore_color.rgb = RGBColor(2, 117, 216)
            elif "Yellow" in color_name or "Orange" in color_name:
                fill.fore_color.rgb = RGBColor(240, 173, 78)
            else:
                fill.fore_color.rgb = RGBColor(91, 192, 222)

    # 2. Add native movable text boxes on the right side for each description/sentence
    text_left = Inches(8.0)
    text_top = Inches(1.0)
    text_width = Inches(4.5)

    for color_name in color_order:
        if color_name not in color_parts_map:
            continue
        
        sentence = color_sentences.get(color_name, f"Details for {color_name}")

        # Create native PowerPoint text box (fully movable & animatable)
        box = slide.shapes.add_textbox(text_left, text_top, text_width, Inches(1.2))
        tf = box.text_frame
        tf.word_wrap = True

        p_head = tf.paragraphs[0]
        p_head.text = f"◆ {color_name} Component"
        p_head.font.size = Pt(14)
        p_head.font.bold = True
        p_head.font.color.rgb = RGBColor(30, 144, 255)

        p_desc = tf.add_paragraph()
        p_desc.text = sentence
        p_desc.font.size = Pt(12)
        p_desc.font.color.rgb = RGBColor(60, 60, 60)

        text_top += Inches(1.3)

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

        st.sidebar.header("⚙️ Native Transition Configuration")
        color_sentences = {}

        if color_parts_map:
            st.sidebar.write("Customize text annotations per shape:")
            color_order = list(color_parts_map.keys())

            for color_name, parts in color_parts_map.items():
                with st.sidebar.expander(f"Color: {color_name} ({len(parts)} elements)"):
                    sentence_input = st.text_input(
                        "Transition / Explanation Note",
                        value=f"Description and animation notes for {color_name}.",
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
            st.subheader("Detected Native Elements Preview")
            preview = img.copy()
            total_elements = 0

            for color_name, parts in color_parts_map.items():
                total_elements += len(parts)
                for p in parts:
                    xmin, ymin, xmax, ymax = p["bbox"]
                    cv2.rectangle(preview, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

            st.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.caption(f"Mapped {total_elements} native editable elements.")

        st.markdown("---")

        render_btn = st.button("📊 Export Native Editable PowerPoint (.pptx)", type="primary", disabled=(len(color_parts_map) == 0))

        if render_btn:
            with st.spinner("Generating native PowerPoint shapes and text boxes..."):
                ppt_bytes = generate_native_editable_powerpoint(img, color_parts_map, color_sentences, color_order)

            st.subheader("📁 Presentation Ready")
            st.success("PowerPoint generated successfully! Every shape and text box is now native and ready for transitions/animations.")

            st.download_button(
                label="Download Native PowerPoint Presentation (.pptx)",
                data=ppt_bytes,
                file_name="native_animatable_diagram.pptx",
                mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
            )

    except Exception as e:
        st.error(f"Error encountered: {e}")
