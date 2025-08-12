
import os
import cv2
import tempfile
import numpy as np
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="occlusionX – Manual POC (No YOLO)", layout="wide")

PRIMARY = "#0A4D8C"
ACCENT = "#4FB3FF"

def header():
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:12px;">
            <div style="width:12px;height:28px;background:{ACCENT};border-radius:4px;"></div>
            <h2 style="margin:0;">occlusionX · Manual Occlusion (No YOLO)</h2>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.caption("Upload a short clip, draw boxes around players, and render an occluded video.")

def apply_style(frame_bgr, boxes, mode="keep_visible", style="dim", alpha=0.65, blur_ksize=31):
    """Apply occlusion based on user-drawn boxes.
    mode = 'keep_visible' -> dim/blur everything OUTSIDE the boxes
    mode = 'occlude_marked' -> dim/blur the INSIDE of the boxes
    """
    out = frame_bgr.copy()

    if len(boxes) == 0:
        return out

    H, W = out.shape[:2]

    if mode == "keep_visible":
        # Create full overlay, then punch out the kept regions
        if style == "blur":
            k = max(3, blur_ksize | 1)
            overlay = cv2.GaussianBlur(out, (k, k), 0)
        else:
            overlay = np.zeros_like(out)
        mask = np.zeros((H, W), dtype=np.uint8)
        for (x1,y1,x2,y2) in boxes:
            x1,y1 = max(0,int(x1)), max(0,int(y1))
            x2,y2 = min(W-1,int(x2)), min(H-1,int(y2))
            mask[y1:y2, x1:x2] = 255
        if style == "blur":
            out = overlay
            # Paste original regions back
            out[mask==255] = frame_bgr[mask==255]
        else:
            # Dim everything, then paste original back into kept boxes
            dimmed = out.copy()
            black = np.zeros_like(out)
            cv2.addWeighted(black, alpha, dimmed, 1-alpha, 0, dimmed)
            dimmed[mask==255] = frame_bgr[mask==255]
            out = dimmed

    else:  # occlude_marked
        for (x1,y1,x2,y2) in boxes:
            x1,y1 = max(0,int(x1)), max(0,int(y1))
            x2,y2 = min(W-1,int(x2)), min(H-1,int(y2))
            roi = out[y1:y2, x1:x2]
            if roi.size == 0: 
                continue
            if style == "blur":
                k = max(3, blur_ksize | 1)
                out[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)
            else:
                black = np.zeros_like(roi)
                cv2.addWeighted(black, alpha, roi, 1-alpha, 0, roi)
                out[y1:y2, x1:x2] = roi

    return out

def write_video(frames_bgr, fps, out_path):
    if not frames_bgr:
        return None
    h, w = frames_bgr[0].shape[:2]
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), max(1, int(fps)), (w,h))
    for f in frames_bgr:
        writer.write(f)
    writer.release()
    return out_path

def parse_canvas_objects(objs):
    """Return list of (x1,y1,x2,y2) from canvas object JSON."""
    boxes = []
    if not objs:
        return boxes
    for o in objs:
        # Only rectangles are supported in this simple POC
        if o.get("type") != "rect":
            # convert circles/others into bounding rect if needed
            left = o.get("left", 0); top = o.get("top", 0)
            w = o.get("width", 0); h = o.get("height", 0)
            boxes.append((left, top, left+w, top+h))
            continue
        left = o.get("left", 0)
        top = o.get("top", 0)
        w = o.get("width", 0)
        h = o.get("height", 0)
        boxes.append((left, top, left+w, top+h))
    return boxes

def main():
    header()

    with st.sidebar:
        st.subheader("Controls")
        sport = st.selectbox("Sport (for labeling)", ["Soccer", "Basketball", "Hockey"], index=0)
        mode = st.radio("Occlusion Mode", ["keep_visible", "occlude_marked"], index=0,
                        help="keep_visible: dim/blur everything except your boxes. occlude_marked: dim/blur inside your boxes.")
        style = st.radio("Style", ["dim", "blur"], index=0)
        alpha = st.slider("Dim strength", 0.10, 0.90, 0.65, 0.05)
        blur = st.slider("Blur kernel (odd)", 3, 51, 31, 2)
        frame_skip = st.slider("Process every Nth frame", 1, 5, 2, 1)
        preview_only = st.checkbox("Preview first frame only", value=True)
        st.markdown("---")
        st.caption("Tip: Draw rectangles around players. Keep clips short (≤20s).")

    col1, col2 = st.columns([1,1])
    with col1:
        video = st.file_uploader("Upload MP4/MOV", type=["mp4","mov","m4v"])

    if not video:
        st.info("Upload a short clip to begin.")
        return

    tdir = tempfile.mkdtemp()
    in_path = os.path.join(tdir, "input.mp4")
    with open(in_path, "wb") as f:
        f.write(video.read())

    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        st.error("Could not read the video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ok, frame0 = cap.read()
    if not ok:
        st.error("Could not read the first frame.")
        return

    frame_rgb = cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB)
    h, w = frame_rgb.shape[:2]

    st.write("**Draw boxes on the first frame**")
    canvas = st_canvas(
        fill_color="rgba(79,179,255,0.2)",
        stroke_width=3,
        stroke_color=ACCENT,
        background_color="#000000",
        background_image=Image.fromarray(frame_rgb),
        width=w, height=h, drawing_mode="rect",
        key="ocx-canvas", update_streamlit=True
    )

    boxes = parse_canvas_objects(canvas.json_data["objects"] if canvas.json_data else [])
    with col2:
        st.write("**Preview with current boxes**")
        prev = apply_style(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR), boxes, mode=mode, style=style, alpha=alpha, blur_ksize=blur)
        st.image(cv2.cvtColor(prev, cv2.COLOR_BGR2RGB), use_container_width=True)

    if preview_only:
        st.warning("Preview-only mode is ON. Uncheck to render the full occluded video.")
        return

    # Process full video
    st.markdown("### Rendering video…")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frames = []
    i = 0
    progress = st.progress(0)
    to_proc = max(1, total // frame_skip)

    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i % frame_skip != 0:
            i += 1
            continue
        out = apply_style(f, boxes, mode=mode, style=style, alpha=alpha, blur_ksize=blur)
        frames.append(out)
        i += 1
        progress.progress(min(1.0, len(frames)/to_proc))

    cap.release()

    out_path = os.path.join(tdir, "occlusionx_manual.mp4")
    write_video(frames, max(1, int(fps // frame_skip)), out_path)

    st.success("Done!")
    st.video(out_path)
    with open(out_path, "rb") as f:
        st.download_button("Download occluded video (MP4)", f, file_name="occlusionx_manual.mp4", mime="video/mp4")

if __name__ == "__main__":
    main()
