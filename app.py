
import os, tempfile, numpy as np, streamlit as st, imageio
from PIL import Image, ImageFilter, ImageDraw

st.set_page_config(page_title="occlusionX – Ultra-Compatible (GIF-only)", layout="wide")

ACCENT = "#4FB3FF"

def header():
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:12px;">
            <div style="width:12px;height:28px;background:{ACCENT};border-radius:4px;"></div>
            <h2 style="margin:0;">occlusionX · Manual Occlusion (No extra deps)</h2>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.caption("Upload a short clip, define rectangles via sliders, and render an animated GIF. No OpenCV/YOLO/Canvas.")

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
            dim = Image.new("RGBA", (W, H), (0,0,0,int(alpha*255)))
            out = img.convert("RGBA")
            out = Image.alpha_composite(out, dim)
            for (x1,y1,x2,y2) in boxes:
                crop = img.crop((x1,y1,x2,y2))
                out.paste(crop, (int(x1),int(y1)))
            return out.convert("RGB")
    else:
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
        max_frames = st.slider("Limit processed frames", 20, 300, 120, 10)
        st.caption("Tip: Short clips (≤20s) render faster.")

    video = st.file_uploader("Upload MP4/MOV", type=["mp4","mov","m4v"])
    if not video:
        st.info("Upload a clip to begin.")
        return

    tdir = tempfile.mkdtemp()
    src_path = os.path.join(tdir, "input.mp4")
    with open(src_path, "wb") as f: f.write(video.read())

    # Read first frame
    try:
        reader = imageio.get_reader(src_path)
        first = reader.get_data(0)
    except Exception as e:
        st.error(f"Could not read video: {e}")
        return

    img = Image.fromarray(first)
    W, H = img.size

    st.write("**Define up to 3 rectangles (x1,y1,x2,y2)**")
    def rect_ui(i):
        st.write(f"Rectangle {i+1}")
        x1 = st.slider(f"x1_{i}", 0, W-1, int(W*0.2), key=f"x1_{i}")
        y1 = st.slider(f"y1_{i}", 0, H-1, int(H*0.2), key=f"y1_{i}")
        x2 = st.slider(f"x2_{i}", 0, W-1, int(W*0.5), key=f"x2_{i}")
        y2 = st.slider(f"y2_{i}", 0, H-1, int(H*0.5), key=f"y2_{i}")
        return (min(x1,x2), min(y1,y2), max(x1,x2), max(y1,y2))

    use_boxes = []
    use_count = st.selectbox("How many rectangles?", [0,1,2,3], index=2)
    for i in range(use_count):
        use_boxes.append(rect_ui(i))

    # Preview
    st.write("**Preview**")
    prev = apply_occlusion_pil(img, use_boxes, mode=mode, style=style, alpha=alpha, blur_radius=blur)
    st.image(prev, use_container_width=True)

    if not st.button("Render GIF"):
        return

    # Render GIF
    st.markdown("### Rendering (GIF)…")
    gif_path = os.path.join(tdir, "occlusionx.gif")
    frames = []
    processed = 0

    try:
        meta = reader.get_meta_data()
        fps = meta.get("fps", 12)
        nframes = meta.get("nframes", 0)
    except Exception:
        fps, nframes = 12, 0

    with st.progress(0) as prog:
        for i, frame in enumerate(reader):
            if i % frame_skip != 0:
                continue
            out = apply_occlusion_pil(Image.fromarray(frame), use_boxes, mode=mode, style=style, alpha=alpha, blur_radius=blur)
            frames.append(out.convert("P", palette=Image.ADAPTIVE))
            processed += 1
            if processed >= max_frames:
                break
            if nframes:
                prog.progress(min(1.0, (i+1)/max(1,nframes)))

    reader.close()
    if not frames:
        st.error("No frames processed. Try different frame skip or clip.")
        return

    try:
        frames[0].save(gif_path, save_all=True, append_images=frames[1:], optimize=True, duration=int(1000/max(1,int(fps/frame_skip))), loop=0)
    except Exception as e:
        st.error(f"Failed to save GIF: {e}")
        return

    st.success("Done!")
    st.image(gif_path)
    with open(gif_path, "rb") as f:
        st.download_button("Download occluded animation (GIF)", f, file_name="occlusionx.gif", mime="image/gif")

if __name__ == "__main__":
    main()
