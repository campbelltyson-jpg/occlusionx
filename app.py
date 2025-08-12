
import os, tempfile, numpy as np, streamlit as st, imageio
from PIL import Image, ImageFilter, ImageDraw
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="occlusionX – Pillow POC (No OpenCV/YOLO)", layout="wide")

ACCENT = "#4FB3FF"

def header():
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:12px;">
            <div style="width:12px;height:28px;background:{ACCENT};border-radius:4px;"></div>
            <h2 style="margin:0;">occlusionX · Manual Occlusion (Pillow + imageio)</h2>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.caption("Upload a short clip, draw rectangles on the first frame, and render an occluded video. No OpenCV/YOLO.")

def apply_occlusion_pil(img: Image.Image, boxes, mode="keep_visible", style="dim", alpha=0.65, blur_radius=12) -> Image.Image:
    W, H = img.size
    if not boxes:
        return img.copy()
    if mode == "keep_visible":
        if style == "blur":
            overlay = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            out = overlay.copy()
            mask = Image.new("L", (W, H), 0)
            mdraw = ImageDraw.Draw(mask)
            for (x1, y1, x2, y2) in boxes:
                mdraw.rectangle([x1, y1, x2, y2], fill=255)
            out.paste(img, mask=mask)
            return out
        else:
            # dim everything except boxes
            dim = Image.new("RGBA", (W, H), (0,0,0,int(alpha*255)))
            out = img.convert("RGBA")
            out = Image.alpha_composite(out, dim)
            # paste original inside boxes
            for (x1,y1,x2,y2) in boxes:
                crop = img.crop((x1,y1,x2,y2))
                out.paste(crop, (int(x1),int(y1)))
            return out.convert("RGB")
    else:
        # occlude marked regions
        out = img.copy()
        for (x1,y1,x2,y2) in boxes:
            region = out.crop((x1,y1,x2,y2))
            if style == "blur":
                region = region.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            else:
                dim_overlay = Image.new("RGBA", region.size, (0,0,0,int(alpha*255)))
                region = region.convert("RGBA")
                region = Image.alpha_composite(region, dim_overlay).convert("RGB")
            out.paste(region, (int(x1),int(y1)))
        return out

def parse_canvas_rects(objs):
    boxes = []
    if not objs: return boxes
    for o in objs:
        left = o.get("left", 0); top = o.get("top", 0)
        w = o.get("width", 0); h = o.get("height", 0)
        boxes.append((left, top, left+w, top+h))
    return boxes

def main():
    header()

    with st.sidebar:
        st.subheader("Controls")
        sport = st.selectbox("Sport label", ["Soccer","Basketball","Hockey"], index=0)
        mode = st.radio("Occlusion Mode", ["keep_visible","occlude_marked"], index=0)
        style = st.radio("Style", ["dim","blur"], index=0)
        alpha = st.slider("Dim strength", 0.10, 0.90, 0.65, 0.05)
        blur = st.slider("Blur radius", 2, 24, 12, 1)
        frame_skip = st.slider("Process every Nth frame", 1, 5, 2, 1)
        preview_only = st.checkbox("Preview first frame only", value=True)
        st.caption("Tip: Short clips (≤20s) render faster.")

    col1, col2 = st.columns([1,1])
    with col1:
        video = st.file_uploader("Upload MP4/MOV", type=["mp4","mov","m4v"])

    if not video:
        st.info("Upload a short clip to begin.")
        return

    tdir = tempfile.mkdtemp()
    src_path = os.path.join(tdir, "input.mp4")
    with open(src_path, "wb") as f: f.write(video.read())

    # Read first frame via imageio
    try:
        reader = imageio.get_reader(src_path, "ffmpeg")
    except Exception as e:
        st.error(f"Could not open video: {e}")
        return

    meta = reader.get_meta_data()
    fps = meta.get("fps", 25)
    nframes = meta.get("nframes", 0)
    st.caption(f"FPS: {fps:.1f} • Frames (reported): {nframes}")

    try:
        first = reader.get_data(0)  # numpy array HxWxC
    except Exception as e:
        st.error(f"Could not read first frame: {e}")
        return

    frame_img = Image.fromarray(first)
    W, H = frame_img.size

    st.write("**Draw rectangles on the first frame**")
    canvas = st_canvas(
        fill_color="rgba(79,179,255,0.2)",
        stroke_width=3,
        stroke_color=ACCENT,
        background_image=frame_img,
        width=W, height=H, drawing_mode="rect",
        key="ocx-canvas", update_streamlit=True
    )

    boxes = parse_canvas_rects(canvas.json_data["objects"] if canvas.json_data else [])
    with col2:
        st.write("**Preview with current boxes**")
        prev = apply_occlusion_pil(frame_img, boxes, mode=mode, style=style, alpha=alpha, blur_radius=blur)
        st.image(prev, use_container_width=True)

    if preview_only:
        st.warning("Preview-only mode is ON. Uncheck to render the full occluded video.")
        return

    st.markdown("### Rendering video…")
    out_path = os.path.join(tdir, "occlusionx_pillow.mp4")
    writer = None
    processed = 0

    try:
        writer = imageio.get_writer(out_path, fps=max(1, int(fps//frame_skip)) or 1, format="ffmpeg", codec="libx264")
    except Exception as e:
        st.error(f"Could not initialize video writer: {e}")
        return

    idx = 0
    with st.progress(0) as prog:
        for i, frame in enumerate(reader):
            if i % frame_skip != 0:
                continue
            img = Image.fromarray(frame)
            occluded = apply_occlusion_pil(img, boxes, mode=mode, style=style, alpha=alpha, blur_radius=blur)
            writer.append_data(np.array(occluded))
            processed += 1
            idx += 1
            # Progress is approximate because nframes may be -1 in streaming formats
            if nframes and frame_skip:
                prog.progress(min(1.0, (i+1)/max(1,nframes)))
    reader.close()
    if writer: writer.close()

    st.success("Done!")
    st.video(out_path)
    with open(out_path, "rb") as f:
        st.download_button("Download occluded video (MP4)", f, file_name="occlusionx_pillow.mp4", mime="video/mp4")

if __name__ == "__main__":
    main()
