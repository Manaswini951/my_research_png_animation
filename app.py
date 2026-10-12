import io
import math
import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

st.set_page_config(
    page_title="Scientific Color-Motion Animation Studio",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 Scientific Color-Motion Animation Studio")
st.write(
    "Upload a scientific diagram or illustration. Assign scientific animation behaviors "
    "to each detected color marker (e.g., Fade-in sequence, Label annotation, Pulsing highlight, Structural shift)."
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

ALL_MOTIONS = [
    "None (Static)",
    "Gradual Fade-In (Sequential Assembly)",
    "Subtle Focus Pulse",
    "Linear Translation / Slide",
    "Continuous Slow Rotation",
    "Steady Growth / Expansion",
]


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


def apply_scientific_motion(map_x, map_y, grid_x, grid_y, parts, motion_type, frame_index, total_frames, intensity, image_shape):
    """Applies scientific/technical motion deformations onto given part masks."""
    if motion_type == "None (Static)" or not parts:
        return

    phase = 2.0 * math.pi * float(frame_index) / float(total_frames)
    sine = math.sin(phase)
    h, w = image_shape[:2]

    for part in parts:
        mask = part["mask"]
        cx, cy = part["center"]

        if motion_type == "Subtle Focus Pulse":
            scale_delta = sine * (intensity / 100.0) * 0.15
            dx = (grid_x - cx) * scale_delta
            dy = (grid_y - cy) * scale_delta
            map_x[mask] -= dx[mask]
            map_y[mask] -= dy[mask]

        elif motion_type == "Linear Translation / Slide":
            dx = sine * (intensity * 0.5)
            map_x[mask] -= dx

        elif motion_type == "Continuous Slow Rotation":
            angle = (float(frame_index) / total_frames) * 0.05 * (intensity / 10.0)
            cos_a, sin_a = math.cos(angle), math.sin(angle)
            orig_x = grid_x[mask] - cx
            orig_y = grid_y[mask] - cy
            rot_x = orig_x * cos_a - orig_y * sin_a + cx
            rot_y = orig_x * sin_a + orig_y * cos_a + cy
            map_x[mask] = rot_x
            map_y[mask] = rot_y

        elif motion_type == "Steady Growth / Expansion":
            growth = (float(frame_index) / total_frames) * (intensity * 0.05)
            dx = (grid_x - cx) * growth
            dy = (grid_y - cy) * growth
            map_x[mask] -= dx[mask]
            map_y[mask] -= dy[mask]


def animate_frame_scientific(original, color_parts_map, color_motion_assignments, color_order, frame_index, total_frames, intensity):
    """Combines scientific animations including blank-white start and sequential assembly with captions."""
    h, w = original.shape[:2]
    
    # 1. Blank white background start
    canvas = np.ones_like(original) * 255

    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = grid_x.copy()
    map_y = grid_y.copy()

    # Track cumulative frames for sequential appearance
    accumulated_frames = 0
    active_colors_data = []

    for idx, color_name in enumerate(color_order):
        if color_name not in color_parts_map:
            continue
        parts = color_parts_map[color_name]
        motion = color_motion_assignments.get(color_name, "None (Static)")
        
        # Each color appears sequentially based on order
        item_start_frame = accumulated_frames
        item_duration = total_frames // max(1, len(color_order))
        accumulated_frames += item_duration

        if frame_index >= item_start_frame:
            local_frame_idx = frame_index - item_start_frame
            
            # Apply motion
            apply_scientific_motion(map_x, map_y, grid_x, grid_y, parts, motion, local_frame_idx, total_frames, intensity, (h, w))
            
            # Calculate fade-in alpha if Gradual Fade-In is selected
            alpha = 1.0
            if motion == "Gradual Fade-In (Sequential Assembly)":
                fade_frames = max(1, item_duration // 2)
                alpha = min(1.0, float(local_frame_idx) / float(fade_frames))

            active_colors_data.append((parts, alpha))

    # Perform remapping for active parts
    warped = cv2.remap(original, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    # Composite onto blank white background with proper alpha blending for fade-ins
    final_frame = np.ones_like(original) * 255
    for parts, alpha in active_colors_data:
        for p in parts:
            mask = p["mask"]
            if alpha >= 1.0:
                final_frame[mask] = warped[mask]
            else:
                final_frame[mask] = cv2.addWeighted(np.ones_like(original)[mask] * 255, 1.0 - alpha, warped[mask], alpha, 0)

    return final_frame


def build_gif(frames, duration=50):
    buf = io.BytesIO()
    prepared = [f.convert("P", palette=Image.ADAPTIVE) for f in frames]
    prepared[0].save(buf, format="GIF", save_all=True, append_images=prepared[1:], duration=duration, loop=0)
    return buf.getvalue()


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

        st.sidebar.header("⚙️ Scientific Animation Dashboard")
        color_motion_assignments = {}
        color_sentences = {}

        if color_parts_map:
            st.sidebar.write("Configure motion and appearance text per detected color region:")
            
            color_order = list(color_parts_map.keys())

            for color_name, parts in color_parts_map.items():
                with st.sidebar.expander(f"Color: {color_name} ({len(parts)} regions)"):
                    selected_motion = st.selectbox(
                        "Behavior",
                        ALL_MOTIONS,
                        index=1, # Default to Gradual Fade-In for scientific assembly
                        key=f"motion_{color_name}",
                    )
                    color_motion_assignments[color_name] = selected_motion

                    sentence_input = st.text_input(
                        "Caption / Explanation Sentence",
                        value=f"Showing component: {color_name}",
                        key=f"sentence_{color_name}"
                    )
                    color_sentences[color_name] = sentence_input
        else:
            st.sidebar.warning("No distinct colored markers detected.")
            color_order = []

        intensity = float(st.sidebar.slider("Animation Intensity / Speed", 4, 30, 10))
        pause_duration = st.sidebar.slider("Delay / Pause Time between appearances (ms)", 100, 2000, 500, step=100)

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

        render_btn = st.button("🔬 Render Scientific Animation", type="primary", disabled=(len(color_parts_map) == 0))

        if render_btn:
            with st.spinner("Rendering blank-white start, sequential assembly, and scientific annotations..."):
                frames = []
                total_frames = 24
                for i in range(total_frames):
                    warped = animate_frame_scientific(img, color_parts_map, color_motion_assignments, color_order, i, total_frames, intensity)
                    
                    # Add caption overlay corresponding to active timeline
                    current_color_idx = min(len(color_order) - 1, int(i / (total_frames / max(1, len(color_order)))))
                    active_color = color_order[current_color_idx] if color_order else ""
                    caption_text = color_sentences.get(active_color, "")

                    # Draw text overlay using PIL
                    pil_img = Image.fromarray(np.ascontiguousarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)))
                    draw = ImageDraw.Draw(pil_img)
                    if caption_text:
                        draw.rectangle([(10, h - 45), (w - 10, h - 10)], fill=(240, 240, 240, 230))
                        draw.text((20, h - 38), caption_text, fill=(20, 20, 20))

                    frames.append(pil_img)

                frame_duration = int(pause_duration / max(1, len(color_order)))
                gif_bytes = build_gif(frames, duration=max(40, frame_duration))

            st.subheader("🎬 Scientific Animation Result")
            st.image(gif_bytes, width=600)
            
            summary_str = " → ".join([f"**{c}**: {color_sentences.get(c, '')}" for c in color_order])
            st.caption(f"Sequential Animation Flow: {summary_str}")

            st.download_button(
                "Download Scientific Animation GIF",
                data=gif_bytes,
                file_name="scientific_diagram_animation.gif",
                mime="image/gif",
            )

    except Exception as e:
        st.error(f"Error encountered: {e}")
