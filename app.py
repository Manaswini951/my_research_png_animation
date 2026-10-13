
import io
import math
import base64
import cv2
import numpy as np
import streamlit as st

st.set_page_config(
    page_title="High-Quality Universal Diagram Vectorizer",
    page_icon="🧬",
    layout="wide",
)

st.title("🧬 High-Quality Universal Diagram Vectorizer")
st.write(
    "Upload any scientific diagram or genomic chart. The tool extracts all shapes, text regions, "
    "and layout blocks into a high-fidelity editable SVG file, keeping detected components "
    "independent and movable."
)


# ============================================================
# COMPONENT DETECTION
# ============================================================

def detect_diagram_components(image):
    """
    Detect diagram components using contour hierarchy and watershed
    segmentation to help separate overlapping or touching shapes.
    """
    h, w = image.shape[:2]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Original foreground detection
    _, thresh = cv2.threshold(
        blurred, 240, 255, cv2.THRESH_BINARY_INV
    )

    # Retrieve nested contours instead of external contours only
    contours, hierarchy = cv2.findContours(
        thresh.copy(),
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    components = []

    # First detect ordinary contours, including nested shapes
    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area < 50:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)

        # Skip contours covering almost the entire image
        if bw > 0.98 * w and bh > 0.98 * h:
            continue

        mask = np.zeros((h, w), dtype=np.uint8)

        # Draw only the current contour, not all nested contours.
        cv2.drawContours(mask, [cnt], -1, 255, thickness=cv2.FILLED)

        components.append((mask, x, y, bw, bh))

    # --------------------------------------------------------
    # Additional detection for touching/overlapping components
    # --------------------------------------------------------

    # Connected foreground regions
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        thresh, connectivity=8
    )

    for label in range(1, num_labels):
        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        bw = stats[label, cv2.CC_STAT_WIDTH]
        bh = stats[label, cv2.CC_STAT_HEIGHT]
        area = stats[label, cv2.CC_STAT_AREA]

        if area < 100 or bw < 3 or bh < 3:
            continue

        if bw > 0.98 * w and bh > 0.98 * h:
            continue

        # Work on this connected region only
        region = (labels[y:y + bh, x:x + bw] == label).astype(
            np.uint8
        ) * 255

        # Distance transform: thicker portions become local maxima
        distance = cv2.distanceTransform(
            region, cv2.DIST_L2, 5
        )

        max_distance = float(distance.max())

        if max_distance <= 0:
            continue

        # Identify possible separate shape centers
        _, sure_foreground = cv2.threshold(
            distance,
            max(1.5, 0.40 * max_distance),
            255,
            cv2.THRESH_BINARY,
        )

        sure_foreground = sure_foreground.astype(np.uint8)

        seed_count, seeds = cv2.connectedComponents(
            sure_foreground, connectivity=8
        )

        # Watershed is useful only when multiple seeds are detected
        if seed_count <= 2:
            continue

        # Build watershed markers:
        # 1 = background, 0 = unknown, 2+ = separate foreground seeds
        local_markers = np.zeros(
            (bh, bw), dtype=np.int32
        )

        local_markers[region == 0] = 1
        local_markers[
            (region > 0) & (sure_foreground == 0)
        ] = 0

        local_markers[sure_foreground > 0] = seeds[
            sure_foreground > 0
        ] + 1

        # Watershed requires a three-channel image
        roi = image[y:y + bh, x:x + bw].copy()

        cv2.watershed(roi, local_markers)

        # Add separately segmented regions
        for seed_id in range(2, seed_count + 1):
            piece = (
                (local_markers == seed_id) &
                (region > 0)
            ).astype(np.uint8) * 255

            piece_area = cv2.countNonZero(piece)

            if piece_area < 50:
                continue

            py, px = np.where(piece > 0)

            if len(px) == 0:
                continue

            px1, px2 = int(px.min()), int(px.max()) + 1
            py1, py2 = int(py.min()), int(py.max()) + 1

            full_mask = np.zeros((h, w), dtype=np.uint8)
            full_mask[
                y + py1:y + py2,
                x + px1:x + px2
            ] = piece[py1:py2, px1:px2]

            components.append((
                full_mask,
                x + px1,
                y + py1,
                px2 - px1,
                py2 - py1,
            ))

    return components


# ============================================================
# IMAGE TO SVG
# ============================================================

def process_diagram_to_svg(image):
    """
    Creates an SVG with independent image-based components.
    Uses contour hierarchy and watershed-assisted segmentation.
    """
    h, w = image.shape[:2]

    components = detect_diagram_components(image)

    svg_elements = []

    svg_elements.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'viewBox="0 0 {w} {h}" width="100%" height="100%">'
    )

    svg_elements.append('  <style>')
    svg_elements.append(
        '    .diagram-element { cursor: move; }'
    )
    svg_elements.append(
        '    .diagram-element:hover { opacity: 0.85; }'
    )
    svg_elements.append('  </style>')

    # White background
    svg_elements.append(
        f'  <rect width="{w}" height="{h}" fill="#ffffff" />'
    )

    shape_idx = 1

    for mask, x, y, bw, bh in components:

        pad = 2

        nx = max(0, x - pad)
        ny = max(0, y - pad)
        ex = min(w, x + bw + pad)
        ey = min(h, y + bh + pad)

        nw = ex - nx
        nh = ey - ny

        if nw <= 0 or nh <= 0:
            continue

        roi_bgr = image[ny:ey, nx:ex]

        # Apply the detected component's mask
        roi_mask = mask[ny:ey, nx:ex]

        roi_bgra = cv2.cvtColor(
            roi_bgr, cv2.COLOR_BGR2BGRA
        )

        roi_bgra[roi_mask == 0, 3] = 0

        # Maximum visual fidelity through PNG encoding
        success, encoded_png = cv2.imencode(
            ".png",
            roi_bgra,
            [cv2.IMWRITE_PNG_COMPRESSION, 3],
        )

        if not success:
            continue

        b64_data = base64.b64encode(
            encoded_png.tobytes()
        ).decode("utf-8")

        svg_elements.append(
            f'  <g id="element_{shape_idx}" '
            f'class="diagram-element">'
        )

        svg_elements.append(
            f'    <image x="{nx}" y="{ny}" '
            f'width="{nw}" height="{nh}" '
            f'xlink:href="data:image/png;base64,{b64_data}" />'
        )

        svg_elements.append('  </g>')

        shape_idx += 1

    svg_elements.append('</svg>')

    return "\n".join(svg_elements), len(components)


# ============================================================
# STREAMLIT UI
# ============================================================

uploaded_file = st.file_uploader(
    "Upload scientific diagram or genomic chart",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded_file is not None:
    try:
        file_bytes = np.asarray(
            bytearray(uploaded_file.read()),
            dtype=np.uint8,
        )

        img = cv2.imdecode(
            file_bytes, cv2.IMREAD_COLOR
        )

        if img is None:
            st.error("Invalid image format.")
            st.stop()

        h, w = img.shape[:2]

        if max(h, w) > 1400:
            scale_val = 1400.0 / max(h, w)

            img = cv2.resize(
                img,
                (
                    int(w * scale_val),
                    int(h * scale_val),
                ),
                interpolation=cv2.INTER_AREA,
            )

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Diagram")

            st.image(
                cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                use_container_width=True,
            )

        with col2:
            st.subheader("Improved Segmentation Preview")

            preview = img.copy()

            components = detect_diagram_components(img)

            count = 0

            for mask, x, y, bw, bh in components:
                cv2.rectangle(
                    preview,
                    (x, y),
                    (x + bw, y + bh),
                    (0, 255, 0),
                    1,
                )
                count += 1

            st.image(
                cv2.cvtColor(preview, cv2.COLOR_BGR2RGB),
                use_container_width=True,
            )

            st.caption(
                f"Detected {count} candidate components, "
                "including nested contours and watershed-separated regions."
            )

        st.markdown("---")

        render_btn = st.button(
            "🧬 Generate High-Quality Movable SVG",
            type="primary",
        )

        if render_btn:
            with st.spinner(
                "Detecting components and building SVG output..."
            ):
                svg_data, component_count = process_diagram_to_svg(
                    img
                )

            st.subheader("📁 Vector SVG Package Ready")

            st.success(
                f"SVG created with {component_count} candidate "
                "image-based components."
            )

            st.download_button(
                label="Download High-Quality Movable SVG (.svg)",
                data=svg_data,
                file_name="high_quality_diagram_movable.svg",
                mime="image/svg+xml",
            )

    except Exception as e:
        st.error(f"Error encountered: {e}")

