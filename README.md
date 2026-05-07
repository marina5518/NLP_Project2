# 🤟 Egyptian Sign Language Translator — Streamlit Deployment

## Project Structure

```
esl_app/
├── app.py               ← Streamlit application
├── requirements.txt     ← Python dependencies
└── README.md
```

Your original files (keep them alongside app.py for reference):
```
Cartoon_Avatar.py
Controller_Detector.py
Video_Matching.py
Gemini_API.py
```

---

## What changed from the original scripts

| Original | Streamlit version |
|---|---|
| `cv2.imshow()` preview window | Removed — not supported on servers |
| `cv2.VideoCapture` input prompt | File uploader widget |
| `input()` sentence prompt | Text input widget |
| Hardcoded dataset path | Configurable text input |
| API key in `.env` | Secure password input |
| `cartoon_avatar.mp4` saved to disk | Download button |
| `animation_data.json` saved to disk | Optional download button |

---

## Step 1 — Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open **http://localhost:8501**

> **Note:** `camel-tools` requires extra setup on Windows.  
> Run: `pip install camel-tools` then `camel-data -i all`

---

## Step 2 — Deploy to Streamlit Cloud

1. Push `app.py` and `requirements.txt` to a **GitHub repo**
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → select repo → set main file to `app.py`
4. Click **Deploy**

> ⚠️ The **full pipeline (Tab 2)** needs the local dataset folder, so it only works locally.  
> **Tab 1** (upload a video) works perfectly on Streamlit Cloud.

---

## Step 3 — Two usage modes

### Tab 1 · Quick mode (no API key needed)
1. Upload any sign-language `.mp4`
2. Click **Extract landmarks & render avatar**
3. Watch the cartoon avatar and download the MP4

### Tab 2 · Full pipeline
Requirements:
- Google Gemini API key (get one at [aistudio.google.com](https://aistudio.google.com))
- Local dataset folder (e.g. `C:\...\Final_Dataset`)
- All dependencies installed

Steps:
1. Enter your Gemini API key
2. Set the dataset folder path
3. Type an Egyptian Arabic sentence
4. Click **Translate & animate**

The app will show a 4-step progress:
- Step 1: Gemini translates to ESL gloss
- Step 2: Videos are matched and merged
- Step 3: MediaPipe extracts landmarks
- Step 4: Cartoon avatar is rendered

---

## Troubleshooting

| Error | Fix |
|---|---|
| `ModuleNotFoundError: mediapipe` | `pip install mediapipe` |
| `ModuleNotFoundError: cv2` | `pip install opencv-python-headless` |
| `camel_tools not found` | `pip install camel-tools && camel-data -i all` |
| `moviepy` FFmpeg error | Install FFmpeg: `winget install ffmpeg` (Windows) or `brew install ffmpeg` (Mac) |
| Gemini API error | Check your API key and model name at [aistudio.google.com](https://aistudio.google.com) |
