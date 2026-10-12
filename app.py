import io
import json
import math
import zipfile
import cv2
import numpy as np
import streamlit as st
from PIL import Image

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Scientific Illustration Animation Studio",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 Scientific Illustration Animation Studio")
st.write(
    "Transform static scientific diagrams, genomics maps, molecular illustrations, "
    "and experimental workflows into publication-quality step-by-step animated presentations."
)


# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================
if "objects" not in st.session_state:
    st.session_state.objects = []
if "canvas_settings" not in st.session_state:
    st.session_state.canvas_settings = {
        "bg_type": "White",
        "custom_color": "#FFFFFF",
        "fps": 30,
        "duration": 6.0,
        "mode": "Build from Blank Canvas",
    }
if "project_loaded" not in st.session_state:
    st.session_state.project_loaded = False


# ============================================================
# IMAGE & OBJECT DETECTION UTILITIES
# ============================================================
def load_image_rgba(uploaded_file):
    """Loads image and preserves RGBA / alpha channels accurately."""
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_cv = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
    if img_cv is None:
        return None

    if len(img_cv.shape) == 2:
        img_rgba = cv2.cvtColor(img_cv, cv2.COLOR_GRAY2RGBA)
    elif img_cv.shape[2] == 3:
        img_rgba = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGBA)
    elif img_cv.shape[2] == 4:
        img_rgba = cv2.cvtColor(img_cv, cv2.COLOR_BGRA2RGBA)
    else:
        img_rgba = img_cv

    h, w = img_rgba.shape[:2]
    if max(h, w) > 1200:
        scale = 1200.0 / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img_rgba = cv2.resize(img_rgba, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return img_rgba


def detect_scientific_objects(img_rgba):
    """Detects individual scientific illustration elements (text, arrows, shapes, lines)."""
    h, w = img_rgba.shape[:2]
    rgb = img_rgba[:, :, :3]
    alpha = img_rgba[:, :, 3] if img_rgba.shape[2] == 4 else np.full((h, w), 255, dtype=np.uint8)

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # Thresholding for shapes & lines
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    # Combine with alpha if present
    if np.any(alpha < 255):
        thresh = cv2.bitwise_and(thresh, alpha)

    # Clean noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned, connectivity=8)

    detected_objects = []
    obj_id = 1

    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < 30:  # Ignore tiny speckles
            continue

        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])

        # Extract object mask
        mask = (labels == i).astype(np.uint8) * 255

        # Heuristic classification
        aspect_ratio = float(bw) / float(max(1, bh))
        obj_type = "Shape"
        if aspect_ratio > 6.0 or aspect_ratio < 0.16:
            obj_type = "Line / Arrow"
        elif bw < 60 and bh < 40 and area < 1500:
            obj_type = "Text / Label"
        elif bw > 0.7 * w and bh > 0.7 * h:
            obj_type = "Background / Frame"
            continue

        # Crop RGBA asset for the object
        obj_rgba = np.zeros((bh, bw, 4), dtype=np.uint8)
        sub_mask = mask[y:y+bh, x:x+bw] > 0
        sub_img = img_rgba[y:y+bh, x:x+bw]
        obj_rgba[sub_mask] = sub_img[sub_mask]

        detected_objects.append({
            "id": f"Obj_{obj_id}",
            "name": f"{obj_type} {obj_id}",
            "type": obj_type,
            "bbox": [x, y, bw, bh],
            "rgba": obj_rgba,
            "mask": sub_mask,
            "visible": True,
            "locked": False,
            # Timeline & Animation settings
            "start_time": float((obj_id - 1) * 0.5),
            "duration": 0.8,
            "entrance": "Fade in" if obj_type != "Line / Arrow" else "Draw Line Progressively",
            "emphasis": "None",
            "exit": "Remain Visible",
            "layer_order": obj_id,
        })
        obj_id += 1

    return detected_objects


# ============================================================
# ANIMATION & RENDERING ENGINE
# ============================================================
def render_animation_frame(objects, canvas_settings, current_time, total_duration, width, height):
    """Renders a single frame of the scientific animation based on the timeline."""
    bg_type = canvas_settings["bg_type"]
    custom_color = canvas_settings["custom_color"]
    mode = canvas_settings["mode"]

    # Initialize canvas
    if bg_type == "White":
        canvas = np.full((height, width, 4), 255, dtype=np.uint8)
    elif bg_type == "Transparent":
        canvas = np.zeros((height, width, 4), dtype=np.uint8)
    else:  # Custom color
        hex_c = custom_color.lstrip("#")
        rgb_c = tuple(int(hex_c[i:i+2], 16) for i in (0, 2, 4))
        canvas = np.full((height, width, 4), (rgb_c[0], rgb_c[1], rgb_c[2], 255), dtype=np.uint8)

    # Sort objects by layer order
    sorted_objs = sorted(objects, key=lambda o: o.get("layer_order", 0))

    for obj in sorted_objs:
        if not obj.get("visible", True):
            continue

        start = obj.get("start_time", 0.0)
        dur = obj.get("duration", 0.5)
        entrance = obj.get("entrance", "Fade in")
        exit_eff = obj.get("exit", "Remain Visible")

        # Determine visibility and progress
        if current_time < start:
            if mode == "Build from Blank Canvas":
                continue
            else:
                # In animate existing diagram mode, show fully before start
                progress = 1.0
        elif start <= current_time <= start + dur:
            progress = (current_time - start) / max(0.01, dur)
        else:
            # After entrance duration
            if exit_eff == "Fade Out" and current_time > (total_duration - 1.0):
                progress = max(0.0, 1.0 - (current_time - (total_duration - 1.0)))
            else:
                progress = 1.0

        x, y, bw, bh = obj["bbox"]
        obj_rgba = obj["rgba"].copy()

        # Apply entrance / effect transformations
        alpha_mult = 1.0
        scale_mult = 1.0
        offset_x = 0
        offset_y = 0

        if entrance == "Fade in":
            alpha_mult = np.clip(progress, 0.0, 1.0)
        elif entrance == "Slide in from Left":
            offset_x = int(-100 * (1.0 - np.clip(progress, 0.0, 1.0)))
            alpha_mult = np.clip(progress, 0.0, 1.0)
        elif entrance == "Slide in from Right":
            offset_x = int(100 * (1.0 - np.clip(progress, 0.0, 1.0)))
            alpha_mult = np.clip(progress, 0.0, 1.0)
        elif entrance == "Scale in / Grow":
            scale_mult = np.clip(progress, 0.01, 1.0)
            alpha_mult = np.clip(progress, 0.0, 1.0)
        elif entrance in ["Draw Line Progressively", "Draw Arrow from Tail to Head"]:
            # Crop width progressively
            clip_w = int(bw * np.clip(progress, 0.0, 1.0))
            if clip_w > 0:
                obj_rgba[:, clip_w:] = 0

        # Apply alpha multiplication
        if alpha_mult < 1.0:
            obj_rgba[:, :, 3] = (obj_rgba[:, :, 3].astype(np.float32) * alpha_mult).astype(np.uint8)

        # Place onto canvas with correct bounds checking
        fx = x + offset_x
        fy = y + offset_y

        # Clip coordinates
        x1 = max(0, fx)
        y1 = max(0, fy)
        x2 = min(width, fx + bw)
        y2 = min(height, fy + bh)

        if x1 >= x2 or y1 >= y2:
            continue

        src_x1 = x1 - fx
        src_y1 = y1 - fy
        src_x2 = src_x1 + (x2 - x1)
        src_y2 = src_y1 + (y2 - y1)

        patch = obj_rgba[src_y1:src_y2, src_x1:src_x2]
        if patch.shape[0] == 0 or patch.shape[1] == 0:
            continue

        # Alpha composite
        alpha_s = patch[:, :, 3:4].astype(np.float32) / 255.0
        dest_roi = canvas[y1:y2, x1:x2].astype(np.float32)

        composite = patch.astype(np.float32) * alpha_s + dest_roi * (1.0 - alpha_s)
        canvas[y1:y2, x1:x2] = np.clip(composite, 0, 255).astype(np.uint8)

    return canvas


# ============================================================
# STREAMLIT INTERFACE & WORKFLOW
# ============================================================
uploaded_file = st.file_uploader("Upload Scientific Diagram (PNG, JPG, JPEG, WebP)", type=["png", "jpg", "jpeg", "webp"])

if uploaded_file is not None:
    if not st.session_state.project_loaded:
        img_rgba = load_image_rgba(uploaded_file)
        if img_rgba is not None:
            st.session_state.original_image = img_rgba
            st.session_state.objects = detect_scientific_objects(img_rgba)
            st.session_state.project_loaded = True
            st.success(f"Successfully detected {len(st.session_state.objects)} scientific elements!")

    if st.session_state.project_loaded:
        img_rgba = st.session_state.original_image
        h, w = img_rgba.shape[:2]

        # ========================================================
        # SIDEBAR CONFIGURATION & TIMELINE CONTROLS
        # ========================================================
        st.sidebar.header("⚙️ Studio Settings & Timeline")

        canvas_settings = st.session_state.canvas_settings
        canvas_settings["mode"] = st.sidebar.selectbox(
            "Composition Mode",
            ["Build from Blank Canvas", "Animate Existing Diagram"],
            index=0 if canvas_settings["mode"] == "Build from Blank Canvas" else 1,
        )
        canvas_settings["bg_type"] = st.sidebar.selectbox(
            "Output Background", ["White", "Transparent", "Custom Color"], index=0
        )
        if canvas_settings["bg_type"] == "Custom Color":
            canvas_settings["custom_color"] = st.sidebar.color_picker("Background Color", "#FFFFFF")

        canvas_settings["fps"] = st.sidebar.slider("Frame Rate (FPS)", 15, 60, 30, 5)
        canvas_settings["duration"] = st.sidebar.slider("Total Timeline Duration (s)", 2.0, 15.0, 6.0, 0.5)

        st.sidebar.markdown("---")
        st.sidebar.subheader("📦 Object Layer Panel")

        # Object inspection and editing
        for idx, obj in enumerate(st.session_state.objects):
            with st.sidebar.expander(f"{obj['name']} ({obj['type']})"):
                obj["name"] = st.text_input("Name", obj["name"], key=f"name_{idx}")
                obj["visible"] = st.checkbox("Visible", obj["visible"], key=f"vis_{idx}")
                obj["start_time"] = st.slider("Start Time (s)", 0.0, canvas_settings["duration"], obj["start_time"], 0.1, key=f"start_{idx}")
                obj["duration"] = st.slider("Entrance Duration (s)", 0.1, 3.0, obj["duration"], 0.1, key=f"dur_{idx}")
                
                entrance_options = [
                    "Fade in", "Slide in from Left", "Slide in from Right", 
                    "Scale in / Grow", "Draw Line Progressively", "Draw Arrow from Tail to Head", "Instant"
                ]
                obj["entrance"] = st.selectbox("Entrance Effect", entrance_options, index=entrance_options.index(obj["entrance"]) if obj["entrance"] in entrance_options else 0, key=f"ent_{idx}")
                
                exit_options = ["Remain Visible", "Fade Out"]
                obj["exit"] = st.selectbox("Exit Effect", exit_options, index=exit_options.index(obj["exit"]) if obj["exit"] in exit_options else 0, key=f"exit_{idx}")
                
                obj["layer_order"] = st.number_input("Stacking Order", value=obj["layer_order"], key=f"layer_{idx}")

                if st.button(f"🗑️ Delete Object {obj['id']}", key=f"del_{idx}"):
                    st.session_state.objects.pop(idx)
                    st.rerun()

        # ========================================================
        # MAIN PREVIEW & TIMELINE SCRUBBER
        # ========================================================
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("🖼️ Original Diagram & Detections")
            preview_img = cv2.cvtColor(img_rgba[:, :, :3], cv2.COLOR_RGB2BGR)
            for obj in st.session_state.objects:
                x, y, bw, bh = obj["bbox"]
                cv2.rectangle(preview_img, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            st.image(cv2.cvtColor(preview_img, cv2.COLOR_BGR2RGB), use_container_width=True)

        with col2:
            st.subheader("🎬 Scene Live Preview")
            scrub_time = st.slider("Timeline Scrubber (Seconds)", 0.0, canvas_settings["duration"], 0.0, 0.1)
            
            current_frame_rgba = render_animation_frame(
                st.session_state.objects, canvas_settings, scrub_time, canvas_settings["duration"], w, h
            )
            st.image(current_frame_rgba, use_container_width=True)

        st.markdown("---")

        # ========================================================
        # EXPORT & PROJECT MANAGEMENT
        # ========================================================
        st.subheader("📤 Export & Project Options")

        col_exp1, col_exp2, col_exp3 = st.columns(3)

        with col_exp1:
            if st.button("✨ Render & Export Animated GIF", type="primary"):
                with st.spinner("Rendering animation frames..."):
                    fps = canvas_settings["fps"]
                    total_dur = canvas_settings["duration"]
                    total_frames = int(fps * total_dur)
                    frames = []

                    for i in range(total_frames):
                        t = (i / total_frames) * total_dur
                        frame_rgba = render_animation_frame(
                            st.session_state.objects, canvas_settings, t, total_dur, w, h
                        )
                        pil_img = Image.fromarray(frame_rgba).convert("P", palette=Image.ADAPTIVE)
                        frames.append(pil_img)

                    gif_buf = io.BytesIO()
                    frames[0].save(
                        gif_buf, format="GIF", save_all=True, append_images=frames[1:],
                        duration=int(1000 / fps), loop=0
                    )
                    gif_bytes = gif_buf.getvalue()

                st.success("Your animation GIF is ready!")
                st.download_button(
                    "Download GIF Animation",
                    data=gif_bytes,
                    file_name="scientific_animation.gif",
                    mime="image/gif",
                )

        with col_exp2:
            if st.button("📦 Export Frame Sequence (ZIP)"):
                with st.spinner("Generating PNG frames..."):
                    fps = canvas_settings["fps"]
                    total_dur = canvas_settings["duration"]
                    total_frames = int(fps * total_dur)
                    zip_buf = io.BytesIO()

                    with zipfile.ZipFile(zip_buf, "w") as zf:
                        for i in range(total_frames):
                            t = (i / total_frames) * total_dur
                            frame_rgba = render_animation_frame(
                                st.session_state.objects, canvas_settings, t, total_dur, w, h
                            )
                            pil_img = Image.fromarray(frame_rgba)
                            img_byte_arr = io.BytesIO()
                            pil_img.save(img_byte_arr, format="PNG")
                            zf.writestr(f"frame_{i:04d}.png", img_byte_arr.getvalue())

                    zip_bytes = zip_buf.getvalue()

                st.success("Frames ZIP ready!")
                st.download_button(
                    "Download Frames ZIP",
                    data=zip_bytes,
                    file_name="animation_frames.zip",
                    mime="application/zip",
                )

        with col_exp3:
            # Project Save / Load JSON
            project_data = {
                "canvas_settings": canvas_settings,
                "objects": [
                    {k: v for k, v in o.items() if k not in ["rgba", "mask"]}
                    for o in st.session_state.objects
                ]
            }
            json_str = json.dumps(project_data, indent=2)
            st.download_button(
                "💾 Save Project Configuration",
                data=json_str,
                file_name="scientific_project.json",
                mime="application/json",
            )

            uploaded_proj = st.file_uploader("📂 Load Project JSON", type=["json"], key="load_proj")
            if uploaded_proj is not None:
                try:
                    proj_dict = json.load(uploaded_proj)
                    st.session_state.canvas_settings = proj_dict["canvas_settings"]
                    st.success("Project settings loaded successfully!")
                except Exception as e:
                    st.error(f"Failed to load project: {e}")

else:
    st.info("👆 Please upload a scientific diagram image (PNG, JPG, WebP) to begin animating.")
