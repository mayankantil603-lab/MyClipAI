# MyClip AI 2

Turn long videos into short clips. Drop a file or paste a link, press **Generate clips**.

MyClip listens to the video (Whisper), measures audio energy and scene changes, reads viewer
signals when the link has them (YouTube "Most replayed", chat replay, timestamped comments),
and can ask an AI model to pick and title the best moments. Clips are cut, reframed to your
format, captioned and saved to `Videos\MyClipAI\Shorts`.

## New in version 2

**Watch and edit inside the app.** Click any clip to open the player:
- play, seek, previous/next clip (Space, ←/→ 5 s, ↑/↓, Esc)
- **Trim**: `[` sets the start, `]` the end, *Save trimmed copy*
- **Captions**: fix any word, merge or split words, *Save and re-make clip* (only that clip is rendered again)
- **Hook**: change or remove the big line at the top
- **Facecam**: drag the box onto your camera (or *Find it for me*); remembered for that video
- **Compare**: the clip heard by Whisper and by the Hindi model side by side; keep the better one

**On the Create page**
- a live **preview** of your caption style, subtitle language, format and hook on a real frame of your video
- **Hook text** on/off: a short AI-written line at the top for the first seconds
- **Bleep swear words**: Off / Beep / Mute (Hindi and English, both scripts; edit the list in Settings)
- **Framing → Facecam**: your facecam on top, gameplay below, captions at the seam

**Results and Library**: Shorts-style cards with a moving preview on hover, select several to upload or
delete, sort by score or timeline, filter, and **Best of**: the top 5 moments in one 60-second video.

**AI helpers** (Settings → AI helpers, one click each, all optional):
| Helper | What it does | Size |
|---|---|---|
| Voice cleanup (Demucs) | removes music, gunshots and game sound before captions | ~2.5 GB with NVIDIA, ~300 MB without |
| Vision | a vision model in Ollama watches the best moments (kills, wins, fails) | none if you have gemma3; else gemma3:4b, 3.3 GB |
| Face tracking (OpenCV) | the crop follows faces; finds your facecam | ~60 MB |
| Hindi speech model (AI4Bharat IndicWhisper) | for Hindi captions; also used by Compare | ~780 MB |
| Sharper thumbnails (Real-ESRGAN) | sharpest frame, upscaled | ~45 MB |

Each helper can be switched off per job under **Advanced**. The *Best quality* preset uses every
installed helper except the Hindi model, which you turn on yourself after comparing.

**Learn from your views** (Settings): uploads made from Studio are checked 48 hours later. Once 8 have
views, MyClip gently favours the kinds of moments that got watched on your channel. Each signal can
move at most 40 %, and *Reset* forgets it.

**Reduce motion** (Settings) turns off fades, slides and moving previews.

## Setup on Windows (first time)

1. **Install Python 3.11 or 3.12** from python.org. On the first installer screen tick
   **Add python.exe to PATH**.
2. **Unzip** this folder somewhere, for example `C:\Users\you\MyClipAI`.
3. **Double-click `run.bat` once.** It installs everything (a few minutes), adds **MyClip AI** to your
   Desktop and Start menu, and opens the app.

From then on, open **MyClip AI** from the Desktop icon or the Start menu, like any other app. No black
window. To pin it: Start → find MyClip AI → right-click → **Pin to taskbar**. Opening it again while it's
running just brings the window to the front.

Keep the folder where it is: the icons point to it. If you move it, run `Create shortcuts.bat` (or
Settings → *Add to Desktop and Start menu*). If the app ever won't open from the icon, run `run.bat`
once: it repairs the setup. Errors are logged to `%APPDATA%\MyClipAI\logs\myclip.log`
(Settings → *Open log*).

### Use your NVIDIA GPU (recommended)
`run.bat` installs GPU support automatically on NVIDIA PCs. `install-gpu.bat` reinstalls it and runs a GPU test. Transcription then runs on the GPU and uses the
large-v3-turbo model automatically. If the GPU can't be used, MyClip falls back to the CPU on its own.

### Add AI (optional)
Clips work without AI. With AI they're picked better and get titles, descriptions and hashtags.

- **Free, local:** models already on your PC are found automatically (Ollama, LM Studio, Jan,
  llama.cpp, KoboldCpp, vLLM). They appear as one-click buttons under *AI*, with size shown on hover.
  If Ollama is installed but not running, MyClip starts it. New models are picked up within a minute,
  or click **Detect**. Other machines on your network: set `MYCLIP_LOCAL_LLM_URLS`.
- **Cloud (OpenAI, Gemini, Groq, OpenRouter, or any OpenAI-compatible API):** click **Add** next to
  *AI* on the Create page, pick the provider, paste your key, Save.

Keys are stored in Windows Credential Manager, never in files. You can also use environment
variables: `MYCLIP_OPENAI_API_KEY`, `MYCLIP_GEMINI_API_KEY`, `MYCLIP_GROQ_API_KEY`,
`MYCLIP_OPENROUTER_API_KEY`, `MYCLIP_LLM_API_KEY`.

### Bigger models on an 8 GB GPU
MyClip frees Whisper from the GPU before the AI runs, unloads Ollama models when a job ends,
and gives Ollama only as much context as each request needs, so more of the model stays on the GPU.
For extra headroom set these user environment variables and restart Ollama:
`OLLAMA_FLASH_ATTENTION=1` and `OLLAMA_KV_CACHE_TYPE=q8_0`. MoE models like `gpt-oss:20b`
and `qwen3:30b-a3b` run well even when they don't fully fit.

## Using it

- **One click everything:** presets (Best quality, Fast, Offline, or your own), saved AI
  connections, and models are all buttons. Right-click a connection or custom preset to edit or delete it.
- **Advanced:** framing (fit with blurred background, fill, follow action), quality up to 4K,
  GPU/CPU, language, viewer signals, scene detection, output folder.
- **Subtitle language** (default *Auto*): Hindi speech becomes Hinglish ("ye kya ho gaya"), English
  stays English, and a word from any other language gets its English meaning in brackets right after it,
  drawn smaller, like "gracias (thanks)". A whole video in another language gets English subtitles.
  Brackets and fixing English words Whisper wrote in Hindi script ("गेम" -> "game") need an AI model;
  without one, Hindi is still converted to Hinglish by built-in rules. You can also force *Hinglish*,
  *English* or *As spoken*. Titles and hashtags follow the same choice.
- **Caption accuracy** (Advanced, default *Best*): after the clips are chosen, each one is transcribed
  again with Whisper large-v3 on cleaned audio. The spoken language is detected from several parts of
  the video, Hindi/Urdu is always written as Hindi, and Whisper's typical mistakes over game sound
  ("thanks for watching", repeated loops) are removed. *Fast* skips the second listen.
- **Length:** pick a range or **Custom** to type your own minimum and maximum (5 s to 10 min).
- After **Generate clips** the settings fold into a one-line summary and the page scrolls to the progress
  box. **Edit settings** opens them again.
- **Shortcuts:** `Ctrl+O` open file, `Ctrl+L` paste link, `Ctrl+Enter` generate, `Esc` cancel,
  `Ctrl+1/2/3` switch page.
- **Library** keeps every job, including custom Save-to folders: search (`Ctrl+F`), play, copy,
  *Clip again* with the same settings, or delete.
- Each job folder has `clips.json` with scores, titles, descriptions and hashtags.

## Live streams

Open **Live**, paste a live link, press **Start watching**.
MyClip records a rolling buffer, learns the stream's normal voice level and chat speed for about
40 seconds, then starts a clip when the streamer suddenly gets loud or chat speeds up (both together
count extra). Clips start ~15 s before the spike and stop once things are back to normal.
Each clip is finished in the background with captions, framing and an AI title.

- **Sensitivity:** Big moments only / Normal / More clips.
- **Make clips:** *While watching* (cards appear live) or *After the stream* (keeps your GPU free while you play).
- **Clip now** (or `Space`) saves the last 30 seconds whenever you want.
- Chat works on Twitch (no login) and YouTube Live. Other sites trigger on voice only.
- Only the last few minutes are kept on disk while watching. Live jobs appear in the Library.

## Studio (YouTube)

Upload clips and manage your channel from the **Studio** page.

- **Upload** on any clip card, or **Upload all** after a job or in the Library. Each clip keeps its AI title,
  description and hashtags. Choose Private / Unlisted / Public, schedule a time, or **Spread out**
  (e.g. one every 24 h).
- Uploads run in a queue in the background, resume after network drops, and continue where they stopped
  if you close the app. If YouTube's daily limit is hit, the queue pauses and tells you.
- See subscribers, views and your recent videos; edit title, description, tags, visibility and thumbnail;
  delete videos; read and reply to recent comments. Several channels: one click to switch.

**One-time setup (about 5 minutes):** Studio → Connect channel walks you through creating a free Google
Cloud "Desktop app" OAuth client and enabling YouTube Data API v3. Your client secret and sign-in
tokens are kept in Windows Credential Manager.

Good to know: new Google projects allow about 6 uploads per day (you can request more), and while
your Google app is "in testing" you may have to reconnect every 7 days. Analytics beyond totals,
monetization and community posts stay in YouTube Studio (there's a button to open it).

## Manual start (if you prefer a terminal)

```
cd path\to\MyClipAI
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-gpu-windows.txt   (optional, NVIDIA)
python main.py
```

## Tests

```
pip install pytest
python -m pytest tests
```

## Project layout

```
main.py                 entry point
myclip/core/            no UI code
  engine.py             the job: import -> analyze -> transcribe -> moments -> AI -> render
  sources.py            files, links (yt-dlp), heatmap, chat replay, comments
  signals.py            audio energy, scene cuts, motion tracking for smart crop
  transcribe.py         faster-whisper with GPU/CPU fallback and transcript cache
  moments.py            signal fusion, candidate windows, scoring, selection
  llm.py                OpenAI-compatible AI: propose, review, titles, model lists
  render.py, captions.py  FFmpeg reframing, captions, thumbnails
  store.py              saved connections (keys in Credential Manager) and presets
myclip/ui/              PySide6 interface
```

## Troubleshooting

- **"Python is not installed"** from run.bat: reinstall Python and tick *Add to PATH*.
- **Models list empty for Ollama:** make sure Ollama is running (it starts with Windows after install).
- **"GPU failed, using CPU"**: the message says why. On NVIDIA PCs `run.bat` installs the GPU libraries
  automatically; `install-gpu.bat` reinstalls/updates them and ends with a GPU test (`python gpu_check.py`).
  Settings → System shows the current GPU status.
- **Slow transcription:** usually the GPU isn't being used (see above), or choose `small` under Transcription.
- Anything else: the error shows in the progress box. Copy it and ask for help.
