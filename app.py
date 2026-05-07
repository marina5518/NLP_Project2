"""
Egyptian Sign Language — Streamlit Web App
==========================================
Converts an Egyptian Arabic sentence → ESL gloss → merged sign video →
MediaPipe landmark extraction → cartoon avatar MP4, all inside the browser.
"""

import sys
import streamlit as st

# ── Python version guard ───────────────────────────────────────────────────────
# mediapipe 0.10.x only ships wheels for Python ≤ 3.12.
# Streamlit Cloud defaults to Python 3.14 which has no mediapipe wheel at all.
# The repo must contain runtime.txt with  python-3.11  to force Python 3.11.
if sys.version_info >= (3, 13):
    st.error(
        f"❌ **Wrong Python version: {sys.version}**\n\n"
        "mediapipe requires Python ≤ 3.12. "
        "Your Streamlit Cloud app is still running Python 3.13+.\n\n"
        "**Fix:** In Streamlit Cloud go to **Manage app → Settings → "
        "Python version** → set to **3.11**, then Reboot app."
    )
    st.stop()

import cv2
import json
import math
import os
import tempfile
import numpy as np
from pathlib import Path
import mediapipe as mp

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Egyptian Sign Language Translator",
    page_icon="🤟",
    layout="wide",
)

# ══════════════════════════════════════════════════════════════════════════════
#  INLINE AVATAR RENDERER  (ported from Cartoon_Avatar.py — no cv2.imshow)
# ══════════════════════════════════════════════════════════════════════════════

CANVAS_W, CANVAS_H = 720, 860
PLAYBACK_FPS       = 30
BODY_CANONICAL_H   = 640
ANCHOR_NOSE_Y      = 129

# Colours (BGR)
SKIN       = (110, 162, 205); SKIN_DARK  = (75,  118, 160); SKIN_LIGHT = (148, 200, 235)
SHIRT      = (65,  105, 180); SHIRT_DARK = (42,   68, 130); SHIRT_LIGHT= (100, 145, 215)
OUTLINE    = (22,   28,  38); HAIR       = (28,   22,  16)
EYE_WHITE  = (245, 245, 240); IRIS       = (55,   38,  26); PUPIL      = (10,    6,   4)
EYE_SHINE  = (255, 255, 255); LIP        = (88,  105, 185)
BG_TOP     = (18,   16,  30); BG_BOT     = (28,   24,  44); TEETH      = (240, 240, 235)

FINGER_SEGS = [
    [(0,1,8),(1,2,7),(2,3,6),(3,4,5)],
    [(5,6,6),(6,7,5),(7,8,4)],
    [(9,10,7),(10,11,5),(11,12,4)],
    [(13,14,6),(14,15,5),(15,16,4)],
    [(17,18,5),(18,19,4),(19,20,3)],
]

def _make_bg(w, h):
    bg  = np.zeros((h, w, 3), dtype=np.uint8)
    top = np.array(BG_TOP, dtype=np.float32)
    bot = np.array(BG_BOT, dtype=np.float32)
    for y in range(h):
        t = y / max(h - 1, 1)
        bg[y] = (top * (1 - t) + bot * t).astype(np.uint8)
    return bg

BG = _make_bg(CANVAS_W, CANVAS_H)

def _get_px(lms, i, w, h):
    if not lms or i >= len(lms): return None
    lm = lms[i]
    if lm.get("_pixel"): return (int(lm["x"]), int(lm["y"]))
    return (int(lm["x"] * w), int(lm["y"] * h))

def _get_z(lms, i):
    if not lms or i >= len(lms): return 0.0
    return lms[i].get("z", 0.0)

def _dist2(p1, p2):
    if p1 is None or p2 is None: return 0.0
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

def _midpt(p1, p2):
    if p1 is None or p2 is None: return None
    return ((p1[0]+p2[0])//2, (p1[1]+p2[1])//2)

def _perp_rect(p1, p2, r):
    dx, dy = p2[0]-p1[0], p2[1]-p1[1]
    d = math.hypot(dx, dy)
    if d < 0.5: return None
    nx, ny = -dy/d*r, dx/d*r
    return np.array([[p1[0]+nx, p1[1]+ny],[p1[0]-nx, p1[1]-ny],
                     [p2[0]-nx, p2[1]-ny],[p2[0]+nx, p2[1]+ny]], dtype=np.int32)

def _perp_unit(p1, p2):
    dx, dy = p2[0]-p1[0], p2[1]-p1[1]
    d = math.hypot(dx, dy)
    if d < 0.5: return (0, 0)
    return (-dy/d, dx/d)

def _capsule(canvas, p1, p2, r, fill, ol, ol_r=3):
    if p1 is None or p2 is None: return
    for pr, color in [(r+ol_r, ol), (r, fill)]:
        poly = _perp_rect(p1, p2, pr)
        if poly is not None: cv2.fillPoly(canvas, [poly], color, cv2.LINE_AA)
        cv2.circle(canvas, p1, pr, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, p2, pr, color, -1, cv2.LINE_AA)

def _capsule_shaded(canvas, p1, p2, r, fill, fill_light, ol):
    _capsule(canvas, p1, p2, r, fill, ol)
    if p1 is None or p2 is None or r < 4: return
    nx, ny = _perp_unit(p1, p2)
    off = r * 0.38
    cv2.line(canvas, (int(p1[0]+nx*off), int(p1[1]+ny*off)),
                     (int(p2[0]+nx*off), int(p2[1]+ny*off)), fill_light, max(1, r//3), cv2.LINE_AA)

def _normalise_pose(pose, cw, ch):
    if not pose or len(pose) < 25: return pose
    nx, ny = pose[0]["x"]*cw, pose[0]["y"]*ch
    hip_y  = ((pose[23]["y"]+pose[24]["y"])/2)*ch
    n2h    = abs(hip_y - ny) or ch*0.5
    scale  = BODY_CANONICAL_H / n2h
    ax, ay = cw/2, float(ANCHOR_NOSE_Y)
    out = []
    for lm in pose:
        out.append({"x":(lm["x"]*cw-nx)*scale+ax, "y":(lm["y"]*ch-ny)*scale+ay,
                    "z":lm.get("z",0), "v":lm.get("v",1), "_pixel":True})
    return out

def _normalise_hand(hand, wrist_px, cw, ch, ref_len):
    if not hand or len(hand)<21: return hand
    wr  = (hand[0]["x"]*cw, hand[0]["y"]*ch)
    mid = (hand[9]["x"]*cw, hand[9]["y"]*ch)
    raw = math.hypot(mid[0]-wr[0], mid[1]-wr[1]) or 1
    s   = (ref_len*0.42)/raw
    return [{"x": wrist_px[0]+(lm["x"]*cw-wr[0])*s,
             "y": wrist_px[1]+(lm["y"]*ch-wr[1])*s,
             "z": lm.get("z",0), "_pixel":True} for lm in hand]

def _analyse_face(face):
    r = {"left_ear":0.25,"right_ear":0.25,"brow_raise":0,"brow_angry":0,"mouth_open":0,"smile":0}
    if not face or len(face)<400: return r
    def fp(i): return (face[i]["x"],face[i]["y"]) if i<len(face) else None
    lc, rc = fp(234), fp(454)
    fw = math.hypot(rc[0]-lc[0],rc[1]-lc[1]) if lc and rc else 0.15 or 0.15
    def df(a,b): return math.hypot(a[0]-b[0],a[1]-b[1])/fw if a and b else 0
    for ear_key, ti,bi,ti2,bi2,inn,out in [
        ("left_ear",159,145,158,153,133,33),
        ("right_ear",386,374,385,380,362,263)]:
        pts = [fp(i) for i in [ti,bi,ti2,bi2,inn,out]]
        if all(pts):
            r[ear_key] = (df(pts[0],pts[1])+df(pts[2],pts[3]))/(2*df(pts[4],pts[5])+1e-5)
    lb,le2,rb,re2 = fp(105),fp(33),fp(334),fp(263)
    if all([lb,le2,rb,re2]):
        avg = (df(lb,le2)+df(rb,re2))/2
        r["brow_raise"] = max(0, min(1,(avg-0.28)/0.15))
    lbi,rbi,nb = fp(55),fp(285),fp(6)
    if all([lbi,rbi,nb]):
        avg = (df(lbi,nb)+df(rbi,nb))/2
        r["brow_angry"] = max(0, min(1,(0.20-avg)/0.10))
    mt,mb,ml,mr = fp(13),fp(14),fp(61),fp(291)
    if all([mt,mb,ml,mr]):
        mh, mw2 = df(mt,mb), df(ml,mr)
        r["mouth_open"] = max(0,min(1,(mh/(mw2+1e-5)-0.05)/0.35))
        mc_y = (mt[1]+mb[1])/2
        r["smile"] = max(0,min(1,((mc_y-ml[1]+mc_y-mr[1])/(2*fw+1e-5))/0.06))
    return r

def render_frame(fd, w=CANVAS_W, h=CANVAS_H):
    canvas = BG.copy()
    pose   = _normalise_pose(fd.get("pose",[]), w, h)
    lhand  = fd.get("left_hand",[])
    rhand  = fd.get("right_hand",[])
    face   = fd.get("face",[])
    G = lambda i: _get_px(pose, i, w, h)
    ls,rs,le,re,lw,rw = G(11),G(12),G(13),G(14),G(15),G(16)
    sw = _dist2(ls,rs) or BODY_CANONICAL_H
    r_up = max(10,int(sw*0.115)); r_lo = max(8,int(sw*0.092))
    fll = _dist2(le,lw) or sw*0.9; flr = _dist2(re,rw) or sw*0.9
    if lw: lhand = _normalise_hand(lhand, lw, w, h, fll)
    if rw: rhand = _normalise_hand(rhand, rw, w, h, flr)
    expr = _analyse_face(face)
    lz, rz = _get_z(pose,15), _get_z(pose,16)
    arms = sorted([(ls,le,lw,lz),(rs,re,rw,rz)], key=lambda a:a[3], reverse=True)
    for sh,el,wr,_ in arms: _capsule_shaded(canvas,sh,el,r_up,SHIRT,SHIRT_LIGHT,OUTLINE); _capsule_shaded(canvas,el,wr,r_lo,SKIN,SKIN_LIGHT,OUTLINE)
    # torso
    if None not in (ls,rs,G(23),G(24)):
        lh2,rh2 = G(23),G(24)
        pts = np.array([ls,rs,rh2,lh2],dtype=np.float32)
        ctr = pts.mean(0)
        cv2.fillPoly(canvas,[((pts-ctr)*1.06+ctr).astype(np.int32)],OUTLINE,cv2.LINE_AA)
        cv2.fillPoly(canvas,[pts.astype(np.int32)],SHIRT,cv2.LINE_AA)
    near = arms[1]
    _capsule_shaded(canvas,near[0],near[1],r_up,SHIRT,SHIRT_LIGHT,OUTLINE)
    _capsule_shaded(canvas,near[1],near[2],r_lo,SKIN,SKIN_LIGHT,OUTLINE)
    # head
    nose = G(0)
    if nose:
        rx = int(sw*0.38); ry = int(sw*0.47)
        cx,cy = nose[0], nose[1]-int(ry*0.28)
        neck_top=(cx,cy+int(ry*0.80)); neck_bot=_midpt(ls,rs) or (cx,cy+int(ry*1.5))
        _capsule_shaded(canvas,neck_top,neck_bot,max(6,int(rx*0.22)),SKIN,SKIN_LIGHT,OUTLINE)
        cv2.ellipse(canvas,(cx,cy-int(ry*0.18)),(rx+10,int(ry*0.72)),0,-185,5,HAIR,-1,cv2.LINE_AA)
        for s in(-1,1): cv2.ellipse(canvas,(cx+s*(rx-2),cy-int(ry*0.05)),(int(rx*0.18),int(ry*0.55)),0,0,360,HAIR,-1,cv2.LINE_AA)
        for s in(-1,1):
            ex=cx+s*(rx-int(rx*0.14)//2); ear_y=cy+int(ry*0.05)
            cv2.ellipse(canvas,(ex,ear_y),(int(rx*0.14)+3,int(ry*0.20)+3),0,0,360,OUTLINE,-1,cv2.LINE_AA)
            cv2.ellipse(canvas,(ex,ear_y),(int(rx*0.14),int(ry*0.20)),0,0,360,SKIN,-1,cv2.LINE_AA)
        cv2.ellipse(canvas,(cx,cy),(rx+4,ry+4),0,0,360,OUTLINE,-1,cv2.LINE_AA)
        cv2.ellipse(canvas,(cx,cy),(rx,ry),0,0,360,SKIN,-1,cv2.LINE_AA)
        # eyes
        eye_y=cy-int(ry*0.18); erx=int(rx*0.19); ery=int(ry*0.11)
        for s in(-1,1):
            eex=cx+s*int(rx*0.37); ear_v=expr["left_ear"] if s==-1 else expr["right_ear"]
            ero=max(2,int(ery*max(0.05,min(1,ear_v/0.30))))
            cv2.ellipse(canvas,(eex,eye_y),(erx+4,ery+4),0,0,360,OUTLINE,-1,cv2.LINE_AA)
            cv2.ellipse(canvas,(eex,eye_y),(erx+1,ery+1),0,0,360,EYE_WHITE,-1,cv2.LINE_AA)
            ir=max(2,int(ero*0.85))
            cv2.circle(canvas,(eex,eye_y),ir+2,OUTLINE,-1,cv2.LINE_AA)
            cv2.circle(canvas,(eex,eye_y),ir,IRIS,-1,cv2.LINE_AA)
            cv2.circle(canvas,(eex,eye_y),max(1,int(ir*0.52)),PUPIL,-1,cv2.LINE_AA)
        # mouth
        mouth_y=cy+int(ry*0.40); mw2=int(rx*0.37)+int(expr["smile"]*int(rx*0.37)*0.35)
        og=int(expr["mouth_open"]*ry*0.18)
        if og>4:
            cv2.rectangle(canvas,(cx-mw2+4,mouth_y-og//2),(cx+mw2-4,mouth_y+og//2),TEETH,-1)
        cv2.ellipse(canvas,(cx,mouth_y),(mw2,max(3,int(ry*0.065))+og),0,0,180,OUTLINE,3,cv2.LINE_AA)
    # hands
    lhz = _get_z(lhand,0) if lhand else -1
    rhz = _get_z(rhand,0) if rhand else -1
    order = [(lhand,lhz),(rhand,rhz)] if lhz<=rhz else [(rhand,rhz),(lhand,lhz)]
    for hlist,_ in order:
        if not hlist or len(hlist)<21: continue
        px2 = [(int(lm["x"]),int(lm["y"])) if lm.get("_pixel") else (int(lm["x"]*w),int(lm["y"]*h)) for lm in hlist]
        palm_pts = np.array([px2[i] for i in [0,1,5,9,13,17]],dtype=np.float32)
        ctr2=palm_pts.mean(0)
        cv2.fillPoly(canvas,[((palm_pts-ctr2)*1.08+ctr2).astype(np.int32)],OUTLINE,cv2.LINE_AA)
        cv2.fillPoly(canvas,[((palm_pts-ctr2)*0.96+ctr2).astype(np.int32)],SKIN,cv2.LINE_AA)
        for finger in FINGER_SEGS:
            for (a,b,r2) in finger: _capsule(canvas,px2[a],px2[b],r2,SKIN,OUTLINE,ol_r=2)
    return canvas


# ══════════════════════════════════════════════════════════════════════════════
#  VIDEO PIPELINE  (adapted from Controller_Detector + Video_Matching)
# ══════════════════════════════════════════════════════════════════════════════

SMOOTH_WINDOW       = 5
CARRY_FORWARD_THRESH = 5

def smooth_track(frames, key, window=SMOOTH_WINDOW, carry=CARRY_FORWARD_THRESH):
    n = len(frames)
    if carry > 0:
        last_valid, gap_start = None, None
        for i in range(n):
            t = frames[i][key]
            if t:
                if gap_start is not None and (i-gap_start) <= carry and last_valid:
                    for j in range(gap_start, i): frames[j][key] = [dict(lm) for lm in last_valid]
                gap_start = None; last_valid = t
            else:
                if gap_start is None: gap_start = i
        if gap_start and last_valid and (n-gap_start) <= carry:
            for j in range(gap_start, n): frames[j][key] = [dict(lm) for lm in last_valid]
    present = [i for i in range(n) if frames[i][key]]
    if not present: return
    sample = frames[present[0]][key]
    n_lm = len(sample); coord_keys = [k for k in sample[0] if k!="v"]
    half = window//2
    for i in present:
        nbrs = [frames[j][key] for j in range(max(0,i-half),min(n,i+half+1)) if frames[j][key] and len(frames[j][key])==n_lm]
        if len(nbrs)<2: continue
        smoothed = []
        for li in range(n_lm):
            avg = {ck: float(np.mean([nb[li][ck] for nb in nbrs if ck in nb[li]])) for ck in coord_keys}
            if "v" in frames[i][key][li]: avg["v"] = frames[i][key][li]["v"]
            smoothed.append(avg)
        frames[i][key] = smoothed


def extract_landmarks(video_path: str, progress_bar=None) -> list:
    """Extract MediaPipe Holistic landmarks from a video file.

    Uses mp.solutions.holistic which is available in mediapipe 0.10.x on
    Python ≤ 3.11.  Streamlit Cloud must be pinned to Python 3.11 via a
    .python-version file in the repo root (see README).
    """
    holistic_mod = mp.solutions.holistic
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    all_frames, frame_count = [], 0

    def _lms(lm_list):
        if lm_list is None:
            return []
        return [{"x": lm.x, "y": lm.y, "z": lm.z,
                 "v": getattr(lm, "visibility", 1.0)}
                for lm in lm_list.landmark]

    try:
        with holistic_mod.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as holistic:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = holistic.process(rgb)
                fd = {
                    "frame_number": frame_count,
                    "pose":        _lms(res.pose_landmarks),
                    "face":        _lms(res.face_landmarks),
                    "left_hand":   _lms(res.left_hand_landmarks),
                    "right_hand":  _lms(res.right_hand_landmarks),
                }
                all_frames.append(fd)
                if progress_bar:
                    progress_bar.progress(
                        min(frame_count / total_frames, 1.0),
                        text=f"Extracting landmarks… frame {frame_count}/{total_frames}")
    finally:
        cap.release()

    for key in ("pose", "face", "left_hand", "right_hand"):
        smooth_track(all_frames, key)
    return all_frames


def _make_browser_compatible(src_path: str) -> str:
    """Re-encode a video so browsers can play it inline (moov atom at front).
    Returns a new temp file path. Falls back to src_path if ffmpeg unavailable."""
    import subprocess, shutil
    if not shutil.which("ffmpeg"):
        return src_path
    try:
        import tempfile as _tf
        out = _tf.NamedTemporaryFile(suffix="_web.mp4", delete=False).name
        subprocess.run(
            ["ffmpeg", "-y", "-i", src_path,
             "-c:v", "libx264", "-preset", "fast", "-crf", "23",
             "-movflags", "+faststart",          # moov atom at front = instant browser play
             "-c:a", "aac", "-b:a", "128k",
             out],
            check=True, capture_output=True
        )
        return out
    except Exception:
        return src_path   # fall back silently


def render_to_video(all_frames: list, out_path: str, progress_bar=None) -> str:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, PLAYBACK_FPS, (CANVAS_W, CANVAS_H))
    total = len(all_frames)
    for i, fd in enumerate(all_frames):
        frame = render_frame(fd)
        writer.write(frame)
        if progress_bar:
            progress_bar.progress((i+1)/total, text=f"Rendering avatar… frame {i+1}/{total}")
    writer.release()
    return out_path


# ══════════════════════════════════════════════════════════════════════════════
#  STREAMLIT UI
# ══════════════════════════════════════════════════════════════════════════════

st.title("🤟 Egyptian Sign Language Translator")
st.markdown(
    "Upload a sign-language video (or let the pipeline generate one from text), "
    "then watch the **cartoon avatar** replay the signs."
)

tab1, tab2 = st.tabs(["🎬 From Video File", "📝 From Arabic Text (full pipeline)"])

# ── Tab 1: Upload a video directly ────────────────────────────────────────────
with tab1:
    st.subheader("Upload a sign-language video")
    uploaded = st.file_uploader("Choose an MP4 / AVI file", type=["mp4","avi","mov"])

    if uploaded:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(uploaded.read())
            video_path = tmp.name

        st.video(video_path)

        if st.button("▶ Extract landmarks & render avatar", key="render_upload"):
            pb1 = st.progress(0, text="Starting…")
            all_frames = extract_landmarks(video_path, pb1)
            pb1.progress(1.0, text=f"Extracted {len(all_frames)} frames ✓")

            out_path = video_path.replace(".mp4", "_avatar.mp4")
            pb2 = st.progress(0, text="Rendering…")
            render_to_video(all_frames, out_path, pb2)
            pb2.progress(1.0, text="Render complete ✓")

            st.success(f"Done! {len(all_frames)} frames rendered.")
            out_path = _make_browser_compatible(out_path)
            st.video(out_path)
            with open(out_path, "rb") as f:
                st.download_button("⬇ Download avatar MP4", f, file_name="cartoon_avatar.mp4", mime="video/mp4")

            # Save JSON
            json_path = video_path.replace(".mp4","_landmarks.json")
            with open(json_path, "w") as jf:
                json.dump(all_frames, jf)
            with open(json_path, "rb") as jf:
                st.download_button("⬇ Download landmark JSON", jf, file_name="animation_data.json", mime="application/json")


# ── Tab 2: Full pipeline with Gemini + dataset ────────────────────────────────
with tab2:
    st.subheader("Full pipeline: Arabic → ESL gloss → avatar")

    with st.expander("⚙️ Pipeline configuration", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            api_key      = st.text_input("Gemini API key", type="password",
                                         help="Your Google Gemini API key from Google AI Studio")
            gemini_model = st.selectbox("Gemini model",
                                        ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.0-flash"],
                                        help="gemini-2.5-flash is recommended. Older models (2.0-flash-lite, 1.5-*) have been deprecated.")
        with c2:
            dataset_source = st.radio(
                "Dataset source",
                ["📂 Google Drive link (recommended for large datasets)",
                 "☁️ Upload ZIP (up to ~200 MB)",
                 "💻 Local folder path"],
                help="For a 700 MB dataset use Google Drive. Upload ZIP works for smaller datasets. Local path works when running locally.")

    # Dataset input based on chosen source
    dataset_path     = None   # resolved below

    if "Google Drive" in dataset_source:
        st.info(
            "📂 **How to share your dataset from Google Drive:**\n\n"
            "1. Upload your dataset ZIP to Google Drive\n"
            "2. Right-click → **Share** → **Anyone with the link** → **Viewer**\n"
            "3. Click **Copy link** and paste it below\n\n"
            "The app will download and extract it once per session (~1-2 min for 700 MB).",
            icon="ℹ️")
        gdrive_url = st.text_input(
            "Google Drive share link",
            placeholder="https://drive.google.com/file/d/XXXX/view?usp=sharing")

        if gdrive_url and st.button("⬇️ Download & extract dataset", key="dl_dataset"):
            import re, zipfile, gdown

            match = re.search(r"/d/([a-zA-Z0-9_-]+)", gdrive_url)
            if not match:
                st.error("Could not parse Google Drive link. Make sure you copy the full share URL.")
                st.stop()
            file_id = match.group(1)

            _zip_dir = tempfile.mkdtemp()
            zip_path = os.path.join(_zip_dir, "dataset.zip")

            bar = st.progress(0, text="Downloading dataset from Google Drive… (this may take 1-2 min for large files)")
            try:
                gdown.download(id=file_id, output=zip_path, quiet=False)
            except Exception as e:
                st.error(f"Download failed: {e}")
                st.stop()

            bar.progress(0.8, text="Download complete — extracting ZIP…")

            if not zipfile.is_zipfile(zip_path):
                st.error(
                    "The downloaded file is not a valid ZIP. "
                    "Make sure your Google Drive link is set to **Anyone with the link → Viewer**."
                )
                st.stop()

            extract_dir = tempfile.mkdtemp()
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(extract_dir)
            os.remove(zip_path)

            bar.progress(1.0, text="Extraction complete ✓")
            entries = [e for e in os.listdir(extract_dir)
                       if os.path.isdir(os.path.join(extract_dir, e))]
            dataset_path = os.path.join(extract_dir, entries[0]) if entries else extract_dir
            st.session_state["dataset_path"] = dataset_path
            st.success(f"✅ Dataset ready — {len(os.listdir(dataset_path))} word folders found.")

        # Persist across reruns
        if "dataset_path" in st.session_state:
            dataset_path = st.session_state["dataset_path"]
            st.success(f"✅ Dataset already loaded — {len(os.listdir(dataset_path))} word folders.")

    elif "Upload ZIP" in dataset_source:
        st.info(
            "📦 Upload a ZIP of your dataset (works for files up to ~200 MB). "
            "For 700 MB use the Google Drive option above.",
            icon="ℹ️")
        uploaded_zip = st.file_uploader("Upload dataset ZIP", type=["zip"], key="dataset_zip")
        if uploaded_zip:
            import zipfile
            _zip_extract_dir = tempfile.mkdtemp()
            with zipfile.ZipFile(uploaded_zip) as zf:
                zf.extractall(_zip_extract_dir)
            entries = [e for e in os.listdir(_zip_extract_dir)
                       if os.path.isdir(os.path.join(_zip_extract_dir, e))]
            dataset_path = os.path.join(_zip_extract_dir, entries[0]) if entries else _zip_extract_dir
            st.success(f"✅ Dataset extracted — found {len(os.listdir(dataset_path))} word folders.")
    else:
        dataset_path = st.text_input(
            "Dataset path",
            value=r"C:\Users\HomePC\Desktop\Final_Dataset\Final_Dataset",
            help="Absolute path to the folder containing one sub-folder per ESL word")

    sentence = st.text_input("Enter Egyptian Arabic sentence", placeholder="مثلاً: أنا بحب الأكل")

    if st.button("🚀 Translate & animate", key="full_pipeline"):
        if not api_key:
            st.error("Please enter your Gemini API key.")
            st.stop()
        if not dataset_path or not os.path.isdir(dataset_path):
            st.error("Dataset not found. Please upload a ZIP or enter a valid local path.")
            st.stop()
        if not sentence.strip():
            st.error("Please enter a sentence.")
            st.stop()

        # ── Step 1: Gemini translation ─────────────────────────────────────
        with st.status("Step 1/4 — Translating to ESL gloss…", expanded=True) as status:
            try:
                import os as _os, time as _time
                _os.environ["GOOGLE_API_KEY"] = api_key
                from google import genai
                from google.genai import types
                from pydantic import BaseModel, Field

                class ListOfWords(BaseModel):
                    words: list[str] = Field(description="List of words in the sentence.")

                words_available = [f.name for f in Path(dataset_path).iterdir() if f.is_dir()]

                client = genai.Client(api_key=api_key)

                # Retry with exponential backoff to handle 429 / quota errors
                _max_retries = 5
                _resp = None
                for _attempt in range(_max_retries):
                    try:
                        _resp = client.models.generate_content(
                            model=gemini_model,
                            contents=[
                                f'''You are an expert in Egyptian Sign Language (ESL / لغة الإشارة المصرية) linguistics.
Your task is to transform a sentence written in Egyptian Arabic dialect into a
restructured Arabic gloss form that follows ESL grammar rules, making it suitable
for direct sign-by-sign translation. You should map each word to the nearest word
in meaning from the provided words in the following word list:
{",".join(words_available)}.

If a word is not available in the provided word list AND it is not a name, then spell
the word using individual Arabic letters (e.g. "محمد" → ["م","ح","م","د"]).
If a word IS in the word list or has a close match, use the matched word AS-IS — do NOT split it into letters.

## ESL Grammar Rules to Apply (in order):

1. **Time-first ordering**: Move all time expressions (e.g., امبارح، بكرة، دلوقتي،
   السنة دي) to the very beginning of the sentence.

2. **Topic-Comment structure**: Identify the topic (what the sentence is about) and
   place it first (after any time marker), followed by the comment about it.

3. **Drop the copula**: Remove linking verbs like (هو، هي، هم، بيكون، كان) when
   used as "is/are/was/were" connectors — sign languages express this spatially.

4. **Drop articles**: Remove definite articles (الـ) and indefinite markers. Keep
   only the base noun.

5. **Drop prepositions where context is clear**: Remove prepositions like (في، على،
   من، إلى، مع) if the spatial relationship is visually obvious. Keep them only
   if semantically essential.

6. **Drop conjunctions**: Remove connectors like (و، لكن، أو، إن، لأن) between
   clauses. List concepts sequentially instead.

7. When given an Arabic verb (in any conjugated form), extract and return ONLY the
   base form (الماضي للمفرد المذكر الغائب / past tense, 3rd person masculine singular).
   Rules:
   - Always return the verb root in past tense base form
   - Example: هيروح → راح | بيكتب → كتب | سيذهب → ذهب
   - Return ONLY the base verb, no explanation, no punctuation
   - If input is already in base form, return it as-is

8. **Adjectives follow nouns**: Ensure all adjectives come directly after the noun
   they describe.

9. **Move negation to the end**: Place (مش، ماـش، مش عارف، لأ) at the end of the
   sentence, not in the middle.

10. **WH-questions — question word goes last**: Move question words (مين، إيه، فين،
    إمتى، ليه، إزاي) to the END of the sentence.

11. **Yes/No questions**: Keep word order but add (؟ إيه رأيك / صح؟) as a final
    marker to indicate a yes/no question.

12. **Keep only content words**: The output should contain only meaningful content
    words — nouns, verbs, adjectives, and essential adverbs.

## Output Format:
You must return a single valid JSON object with a "words" key containing a list of strings.
Do not include any text, explanation, or markdown outside the JSON.

Now apply these rules to the following Egyptian Arabic sentence:''',
                                sentence
                            ],
                            config={"response_mime_type": "application/json",
                                    "response_json_schema": ListOfWords.model_json_schema()}
                        )
                        break  # success — exit retry loop
                    except Exception as _e:
                        _err_str = str(_e)
                        _is_rate_limit = ("429" in _err_str or "RESOURCE_EXHAUSTED" in _err_str
                                          or "quota" in _err_str.lower())
                        if _is_rate_limit and _attempt < _max_retries - 1:
                            _wait = 2 ** (_attempt + 1)   # 2, 4, 8, 16 sec
                            st.warning(
                                f"⏳ Rate limit hit — waiting {_wait}s before retry "
                                f"(attempt {_attempt + 1}/{_max_retries})…"
                            )
                            _time.sleep(_wait)
                        else:
                            raise   # re-raise on non-rate-limit error or last attempt

                if _resp is None:
                    raise RuntimeError("All retry attempts failed.")

                resp = _resp
                gloss_words = ListOfWords.model_validate_json(resp.text).words
                import re as _re
                def dediac(w): return _re.sub(r"[ً-ٟ]", "", w)
                gloss_words = [dediac(w) for w in gloss_words]
                status.update(label=f"Step 1/4 — Gloss: **{' | '.join(gloss_words)}**", state="complete")
            except Exception as e:
                st.error(f"Gemini error: {e}")
                st.stop()

        # ── Step 2: Video matching ─────────────────────────────────────────
        with st.status("Step 2/4 — Matching sign videos…", expanded=True) as status:
            try:
                import glob
                from moviepy.editor import VideoFileClip, concatenate_videoclips, CompositeVideoClip

                video_paths = []
                for word in gloss_words:
                    word_dir = os.path.join(dataset_path, word)
                    if os.path.isdir(word_dir):
                        vids = glob.glob(os.path.join(word_dir, "*.mp4"))
                        if vids: video_paths.append(vids[0])
                        else: st.warning(f"No MP4 in folder for: {word}")
                    else:
                        st.warning(f"No folder found for word: {word}")

                if not video_paths:
                    st.error("No matching videos found. Check your dataset path and gloss words.")
                    st.stop()

                clips = [VideoFileClip(p) for p in video_paths]
                max_w = max(c.w for c in clips)
                max_h = max(c.h for c in clips)

                def _pad(clip, tw, th):
                    sc = min(tw/clip.w, th/clip.h)
                    s  = clip.resize(sc)
                    xo, yo = (tw-s.w)//2, (th-s.h)//2
                    return CompositeVideoClip([s.set_position((xo,yo))], size=(tw,th)).set_duration(clip.duration)

                resized = [_pad(c, max_w, max_h) for c in clips]
                merged  = concatenate_videoclips(resized, method="compose")

                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                    merged_path = tmp.name
                merged.write_videofile(merged_path, codec="libx264", audio_codec="aac", logger=None)
                for c in clips+resized: c.close()
                merged.close()

                merged_path = _make_browser_compatible(merged_path)
                status.update(label=f"Step 2/4 — Merged {len(video_paths)} clips ✓", state="complete")
                st.video(merged_path)
            except Exception as e:
                st.error(f"Video matching error: {e}")
                st.stop()

        # ── Step 3: MediaPipe extraction ───────────────────────────────────
        with st.status("Step 3/4 — Extracting MediaPipe landmarks…", expanded=True) as status:
            pb = st.progress(0)
            all_frames = extract_landmarks(merged_path, pb)
            status.update(label=f"Step 3/4 — {len(all_frames)} frames extracted ✓", state="complete")

        # ── Step 4: Render avatar ──────────────────────────────────────────
        with st.status("Step 4/4 — Rendering cartoon avatar…", expanded=True) as status:
            pb2 = st.progress(0)
            with tempfile.NamedTemporaryFile(suffix="_avatar.mp4", delete=False) as tmp:
                avatar_path = tmp.name
            render_to_video(all_frames, avatar_path, pb2)
            status.update(label="Step 4/4 — Avatar rendered ✓", state="complete")

        avatar_path = _make_browser_compatible(avatar_path)
        st.success("✅ All done!")
        st.video(avatar_path)
        with open(avatar_path, "rb") as f:
            st.download_button("⬇ Download avatar MP4", f, file_name="cartoon_avatar.mp4", mime="video/mp4")


# ── Sidebar info ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🤟 ESL Translator")
    st.markdown("""
**Tab 1 — Quick mode**
Upload any sign-language MP4 → get the cartoon avatar immediately. No API key needed.

**Tab 2 — Full pipeline**
Requires:
- Google Gemini API key
- ESL dataset as a ZIP upload **or** a local folder path
- `camel-tools`, `moviepy`, `google-genai` installed

**Controls on downloaded video:**
The rendered MP4 loops seamlessly and can be scrubbed in any video player.
""")
    st.divider()
    st.caption("Built with MediaPipe · OpenCV · Streamlit")
