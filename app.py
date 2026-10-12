import io
import json
import math
import zipfile
import cv2
import numpy as np
import streamlit as st
from PIL import Image

# ============================================================
# PAGE CONFIGURATION & SETUP
# ============================================================
st.set_page_config(
    page_title="Scientific Illustration Animation Studio",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 Scientific Illustration Animation Studio")
st.write(
    "Upload research diagrams. Inspect color-coded elements, edit text labels with your "
    "actual diagram sentences, and organize them into professional animation sequences."
)

# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================
if "objects" not in st.session_state:
    st.session_state.objects = []
if "animation_sequence" not in st.session_state:
    st.session_state.animation_sequence = []
if "canvas_settings" not in st.session_state:
    st.session_state.canvas_settings = {
        "mode": "Build from Blank Canvas",
        "bg_type": "White",
        "custom_color": "#FFFFFF",
        "fps": 30,
        "duration": 6.0,
    }
if "project_loaded" not in st.session_state:
    st.session_state.project_loaded = False


# ============================================================
# COLOR CATEGORIZATION & OBJECT EXTRACTION
# ============================================================
def classify_color(rgb_color):
    """Classifies an RGB color tuple into precise scientific palette groups."""
    r, g, b = int(rgb_color[0]), int(rgb_color[1]), int(rgb_color[2])
    
    if r > 200 and g > 200 and b > 200:
        return "White / Light Gray"
    if r < 50 and g < 50 and b < 50:
        return "Black / Dark Gray (Non-colored)"
    
    hsv = cv2.cvtColor(np.uint8([[[r, g, b]]]), cv2.COLOR_RGB2HSV)[0][0]
    h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])

    if s < 40 and v > 50:
        return "White / Light Gray"
    if v < 40:
        return "Black / Dark Gray (Non-colored)"

    if h < 10 or h > 170:
        return "Red"
    elif 10 <= h < 22:
        return "Orange"
    elif 22 <= h < 35:
        return "Yellow"
    elif 35 <= h < 85:
        return "Green"
    elif 85 <= h < 100:
        return "Cyan / Light Blue"
    elif 100 <= h < 130:
        return "Blue"
    elif 130 <= h < 160:
        return "Purple / Violet"
    elif 160 <= h <= 170:
        return "Pink / Magenta"
    
    return "Other Colors"


def load_image_rgba(uploaded_file):
    """Loads uploaded image file and preserves RGBA / alpha channels."""
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
    """Detects and separates individual scientific objects, color regions, and non-colored lines/arrows."""
    h, w = img_rgba.shape[:2]
    rgb = img_rgba[:, :, :3]
    alpha = img_rgba[:, :, 3] if img_rgba.shape[2] == 4 else np.full((h, w), 255, dtype=np.uint8)

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    if np.any(alpha < 255):
        thresh = cv2.bitwise_and(thresh, alpha)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned, connectivity=8)

    detected_objects = []
    obj_id = 1

    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < 25:
            continue

        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])

        if bw > 0.9 * w and bh > 0.9 * h:
            continue

        sub_mask = (labels == i).astype(np.uint8) * 255
        
        component_pixels = rgb[sub_mask > 0]
        if len(component_pixels) > 0:
            mean_color = np.mean(component_pixels, axis=0)
        else:
            mean_color = np.array([0, 0, 0])

        color_category = classify_color(mean_color)

        aspect_ratio = float(bw) / float(max(1, bh))
        if aspect_ratio > 7.0 or aspect_ratio < 0.14:
            obj_type = "Non-colored Line / Arrow" if "Black" in color_category or "White" in color_category else "Colored Connector / Arrow"
        elif bw < 90 and bh < 50 and area < 1500:
            obj_type = "Text Label / Sentence"
        elif 0.8 < aspect_ratio < 1.2 and area < 2500:
            obj_type = "Circle / Marker"
        else:
            obj_type = "Shape / Region"

        obj_rgba = np.zeros((bh, bw, 4), dtype=np.uint8)
        mask_crop = sub_mask[y:y+bh, x:x+bw] > 0
        img_crop = img_rgba[y:y+bh, x:x+bw]
        obj_rgba[mask_crop] = img_crop[mask_crop]

        # Meaningful default naming instead of generic numbering
        if "Text" in obj_type:
            default_name = f"Label {obj_id} (Edit your text here)"
        else:
            default_name = f"{color_category.split(' ')[0]} {obj_type.split('/')[0].strip()} {obj_id}"

        detected_objects.append({
            "id": f"obj_{obj_id}",
            "name": default_name,
            "type": obj_type,
            "color_category": color_category,
            "mean_color": [int(c) for c in mean_color],
            "bbox": [x, y, bw, bh],
            "rgba": obj_rgba,
            "mask": mask_crop,
            "visible": True,
            "locked": False,
        })
        obj_id += 1

    return detected_objects


# ============================================================
# RENDERING ENGINE
# ============================================================
def render_animation_frame(sequence_items, canvas_settings, current_time, total_duration, width, height):
    bg_type = canvas_settings.get("bg_type", "White")
    custom_color = canvas_settings.get("custom_color", "#FFFFFF")
    mode = canvas_settings.get("mode", "Build from Blank Canvas")

    if bg_type == "White":
        canvas = np.full((height, width, 4), 255, dtype=np.uint8)
    elif bg_type == "Transparent":
        canvas = np.zeros((height, width, 4), dtype=np.uint8)
    else:
        hex_c = custom_color.lstrip("#")
        rgb_c = tuple(int(hex_c[i:i+2], 16) for i in (0, 2, 4))
        canvas = np.full((height, width, 4), (rgb_c[0], rgb_c[1], rgb_c[2], 255), dtype=np.uint8)

    for item in sequence_items:
        obj = item["object"]
        start = item.get("start_time", 0.0)
        dur = item.get("duration", 0.8)
        entrance = item.get("entrance", "Fade in")
        exit_eff = item.get("exit", "Remain Visible")

        if current_time < start:
            if mode == "Build from Blank Canvas":
                continue
            else:
                progress = 1.0
        elif start <= current_time <= start + dur:
            progress = (current_time - start) / max(0.01, dur)
        else:
            if exit_eff == "Fade Out" and current_time > (total_duration - 1.0):
                progress = max(0.0, 1.0 - (current_time - (total_duration - 1.0)))
            else:
                progress = 1.0

        x, y, bw, bh = obj["bbox"]
        obj_rgba = obj["rgba"].copy()
        alpha_mult = 1.0
        offset_x = 0

        if entrance == "Fade in":
            alpha_mult = np.clip(progress, 0.0, 1.0)
        elif entrance == "Slide in from Left":
            offset_x = int(-120 * (1.0 - np.clip(progress, 0.0, 1.0)))
            alpha_mult = np.clip(progress, 0.0, 1.0)
        elif entrance == "Slide in from Right":
            offset_x = int(120 * (1.0 - np.clip(progress, 0.0, 1.0)))
            alpha_mult = np.clip(progress, 0.0, 1.0)
        elif entrance in ["Draw Line Progressively", "Draw Arrow from Tail to Head"]:
            clip_w = int(bw * np.clip(progress, 0.0, 1.0))
            if clip_w > 0:
                obj_rgba[:, clip_w:] = 0

        if alpha_mult < 1.0:
            obj_rgba[:, :, 3] = (obj_rgba[:, :, 3].astype(np.float32) * alpha_mult).astype(np.uint8)

        fx = x + offset_x
        fy = y

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

        alpha_s = patch[:, :, 3:4].astype(np.float32) / 255.0
        dest_roi = canvas[y1:y2, x1:x2].astype(np.float32)

        composite = patch.astype(np.float32) * alpha_s + dest_roi * (1.0 - alpha_s)
        canvas[y1:y2, x1:x2] = np.clip(composite, 0, 255).astype(np.uint8)

    return canvas


# ============================================================
# STREAMLIT UI WORKFLOW
# ============================================================
uploaded_file = st.file_uploader("Upload Scientific Diagram (PNG, JPG, JPEG, WebP)", type=["png", "jpg", "jpeg", "webp"])

if uploaded_file is not None:
    if not st.session_state.project_loaded:
        img_rgba = load_image_rgba(uploaded_file)
        if img_rgba is not None:
            st.session_state.original_image = img_rgba
            st.session_state.objects = detect_scientific_objects(img_rgba)
            st.session_state.animation_sequence = []
            st.session_state.project_loaded = True
            st.success(f"Successfully detected {len(st.session_state.objects)} scientific elements!")

    if st.session_state.project_loaded:
        img_rgba = st.session_state.original_image
        h, w = img_rgba.shape[:2]

        st.sidebar.header("⚙️ Studio Settings")
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
        canvas_settings["duration"] = st.sidebar.slider("Total Timeline Duration (s)", 2.0, 20.0, 6.0, 0.5)

        st.sidebar.markdown("---")
        st.sidebar.subheader("✏️ Rename Sentences & Objects")
        st.write("Type your actual diagram text sentences below:")
        
        for obj in st.session_state.objects:
            if "color_category" not in obj:
                obj["color_category"] = "Other Colors"
            with st.sidebar.expander(f"{obj['type']} [{obj['color_category']}]"):
                obj["name"] = st.text_input("Enter real text / name:", obj["name"], key=f"rename_{obj['id']}")
                if st.button(f"🗑️ Delete {obj['id']}", key=f"del_obj_{obj['id']}"):
                    st.session_state.objects = [o for o in st.session_state.objects if o["id"] != obj["id"]]
                    st.session_state.animation_sequence = [item for item in st.session_state.animation_sequence if item["object"]["id"] != obj["id"]]
                    st.rerun()

        tab1, tab2, tab3 = st.tabs([
            "🔍 1. Visual Element Selection & Color Filter", 
            "⏱️ 2. Animation Sequence Builder", 
            "🎬 3. Live Preview & Export"
        ])

        with tab1:
            st.subheader("Visual Element Discovery Dashboard")
            st.write(
                "Elements are separated by **Color Groups** and **Object Types** (including non-colored lines/arrows). "
                "Inspect thumbnails and click **'Add to Animation Sequence'**."
            )

            filter_mode = st.radio("Filter Options By:", ["Color Grouping", "Object Type Grouping"], horizontal=True)

            for obj in st.session_state.objects:
                if "color_category" not in obj:
                    obj["color_category"] = "Other Colors"

            if filter_mode == "Color Grouping":
                colors_found = sorted(list(set(o.get("color_category", "Other Colors") for o in st.session_state.objects)))
                selected_color = st.selectbox("Select Color Group:", colors_found)
                filtered_objs = [o for o in st.session_state.objects if o.get("color_category", "Other Colors") == selected_color]
            else:
                types_found = sorted(list(set(o["type"] for o in st.session_state.objects)))
                selected_type = st.selectbox("Select Object Type:", types_found)
                filtered_objs = [o for o in st.session_state.objects if o["type"] == selected_type]

            st.markdown(f"**Showing {len(filtered_objs)} elements:**")

            cols = st.columns(3)
            for idx, obj in enumerate(filtered_objs):
                with cols[idx % 3]:
                    st.markdown(f"**{obj['name']}**")
                    thumb_rgb = cv2.cvtColor(obj["rgba"], cv2.COLOR_BGRA2RGBA)
                    st.image(thumb_rgb, width=150)
                    
                    already_in_seq = any(item["object"]["id"] == obj["id"] for item in st.session_state.animation_sequence)
                    if not already_in_seq:
                        if st.button(f"➕ Add to Sequence", key=f"add_{obj['id']}"):
                            st.session_state.animation_sequence.append({
                                "object": obj,
                                "start_time": float(len(st.session_state.animation_sequence) * 0.8),
                                "duration": 0.8,
                                "entrance": "Fade in" if "Line" not in obj["type"] else "Draw Line Progressively",
                                "exit": "Remain Visible",
                            })
                            st.success(f"Added {obj['name']} to sequence!")
                            st.rerun()
                    else:
                        st.info("✓ In Animation Sequence")

            st.markdown("---")
            st.subheader("Full Diagram Overview")
            overview_img = cv2.cvtColor(img_rgba[:, :, :3], cv2.COLOR_RGB2BGR)
            for obj in st.session_state.objects:
                x, y, bw, bh = obj["bbox"]
                cv2.rectangle(overview_img, (x, y), (x + bw, y + bh), (0, 0, 255), 2)
            st.image(cv2.cvtColor(overview_img, cv2.COLOR_BGR2RGB), width="stretch")

        with tab2:
            st.subheader("Animation Sequence Builder")
            st.write("Arrange your chosen sentences, colored shapes, and non-colored arrows into your custom animation sequence.")

            if not st.session_state.animation_sequence:
                st.warning("No elements in sequence yet. Add elements from Tab 1!")
            else:
                for idx, item in enumerate(st.session_state.animation_sequence):
                    obj = item["object"]
                    with st.expander(f"Step {idx+1}: {obj['name']} ({obj['type']})", expanded=True):
                        col_a, col_b, col_c = st.columns([1, 2, 1])
                        
                        with col_a:
                            st.image(cv2.cvtColor(obj["rgba"], cv2.COLOR_BGRA2RGBA), width=120)
                        
                        with col_b:
                            item["start_time"] = st.slider(
                                "Start Time (s)", 0.0, canvas_settings["duration"], item["start_time"], 0.1, key=f"seq_start_{idx}"
                            )
                            item["duration"] = st.slider(
                                "Duration (s)", 0.1, 3.0, item["duration"], 0.1, key=f"seq_dur_{idx}"
                            )
                            
                            entrance_options = [
                                "Fade in", "Slide in from Left", "Slide in from Right", 
                                "Draw Line Progressively", "Draw Arrow from Tail to Head", "Instant"
                            ]
                            item["entrance"] = st.selectbox(
                                "Entrance Effect", entrance_options, 
                                index=entrance_options.index(item["entrance"]) if item["entrance"] in entrance_options else 0,
                                key=f"seq_ent_{idx}"
                            )

                            exit_options = ["Remain Visible", "Fade Out"]
                            item["exit"] = st.selectbox(
                                "Exit Effect", exit_options,
                                index=exit_options.index(item["exit"]) if item["exit"] in exit_options else 0,
                                key=f"seq_exit_{idx}"
                            )

                        with col_c:
                            st.markdown("<br>", unsafe_allow_html=True)
                            if idx > 0 and st.button("⬆️ Move Up", key=f"up_{idx}"):
                                st.session_state.animation_sequence[idx], st.session_state.animation_sequence[idx-1] = \
                                    st.session_state.animation_sequence[idx-1], st.session_state.animation_sequence[idx]
                                st.rerun()
                            if idx < len(st.session_state.animation_sequence) - 1 and st.button("⬇️ Move Down", key=f"down_{idx}"):
                                st.session_state.animation_sequence[idx], st.session_state.animation_sequence[idx+1] = \
                                    st.session_state.animation_sequence[idx+1], st.session_state.animation_sequence[idx]
                                st.rerun()
                            if st.button("❌ Remove", key=f"rem_{idx}"):
                                st.session_state.animation_sequence.pop(idx)
                                st.rerun()

        with tab3:
            st.subheader("Interactive Scene Preview & Multi-Format Export")
            
            scrub_time = st.slider(
                "Scrub Timeline Preview (Seconds)", 0.0, canvas_settings["duration"], 0.0, 0.1, key="scrubber"
            )

            current_frame = render_animation_frame(
                st.session_state.animation_sequence, canvas_settings, scrub_time, canvas_settings["duration"], w, h
            )
            st.image(current_frame, width="stretch")

            st.markdown("---")
            st.subheader("📤 Export Options")

            col_exp1, col_exp2, col_exp3 = st.columns(3)

            with col_exp1:
                if st.button("✨ Render & Export Animated GIF", type="primary"):
                    with st.spinner("Rendering animation GIF frames..."):
                        fps = canvas_settings["fps"]
                        total_dur = canvas_settings["duration"]
                        total_frames = int(fps * total_dur)
                        gif_frames = []

                        for i in range(total_frames):
                            t = (i / total_frames) * total_dur
                            frame_rgba = render_animation_frame(
                                st.session_state.animation_sequence, canvas_settings, t, total_dur, w, h
                            )
                            pil_img = Image.fromarray(frame_rgba).convert("P", palette=Image.ADAPTIVE)
                            gif_frames.append(pil_img)

                        gif_buf = io.BytesIO()
                        gif_frames[0].save(
                            gif_buf, format="GIF", save_all=True, append_images=gif_frames[1:],
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
                    with st.spinner("Generating PNG frames ZIP..."):
                        fps = canvas_settings["fps"]
                        total_dur = canvas_settings["duration"]
                        total_frames = int(fps * total_dur)
                        zip_buf = io.BytesIO()

                        with zipfile.ZipFile(zip_buf, "w") as zf:
                            for i in range(total_frames):
                                t = (i / total_frames) * total_dur
                                frame_rgba = render_animation_frame(
                                    st.session_state.animation_sequence, canvas_settings, t, total_dur, w, h
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
                serializable_seq = [
                    {
                        "object_id": item["object"]["id"],
                        "object_name": item["object"]["name"],
                        "start_time": item["start_time"],
                        "duration": item["duration"],
                        "entrance": item["entrance"],
                        "exit": item["exit"],
                    }
                    for item in st.session_state.animation_sequence
                ]
                project_data = {
                    "canvas_settings": canvas_settings,
                    "animation_sequence": serializable_seq,
                }
                json_str = json.dumps(project_data, indent=2)
                st.download_button(
                    "💾 Save Project Configuration",
                    data=json_str,
                    file_name="scientific_animation_project.json",
                    mime="application/json",
                )

else:
    st.info("👆 Please upload a scientific diagram image (PNG, JPG, JPEG, WebP) in Tab 1 or the sidebar to begin.")
