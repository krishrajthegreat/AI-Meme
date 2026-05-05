# AI Meme Emote Detector 🎭

A real-time computer vision desktop app that detects your facial expressions
and hand gestures via webcam and displays a matching brainrot meme.

Built with Python, OpenCV, and MediaPipe.

---

## Demo

Split-screen layout: webcam feed on the left, meme on the right.

---

## Supported Gestures

| Gesture | Trigger |
|---|---|
| Shock 😱 | Both hands on head + jaw drop |
| Shaq T ⏱️ | Timeout T gesture with two hands |
| LeBron 👑 | Both hands low + open mouth scream |
| Giggle 🤭 | Hand covering mouth |
| Shush 🤫 | Finger on lips + sideways face |
| Thinking 🤔 | Finger near mouth + open mouth |
| Cut It Out ✋ | Flat open hand at chin level |
| Self Pointing 👉 | Index finger pointing at camera |
| Jerry 🐭 | Fully spread open hand |
| Smirk 😏 | Asymmetric smile |
| Wink 😉 | One eye closed |
| Speed ⚡ | Squint + pursed lips |
| Patrick ⭐ | Jaw drop, no hands |

---

## Requirements

- Python 3.11+
- Webcam
- Git

---

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/krishrajthegreat/AI-Meme.git
cd AI-Meme

# 2. Create and activate virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your meme images to ./images/
# See the Images section below for required filenames

# 5. Run
python main.py
```

## Controls

| Key | Action |
|---|---|
| `q` | Quit |
| `d` | Toggle debug overlay |

---

## Meme Images

Place these files in the `./images/` folder. MediaPipe models are
downloaded automatically on first run into `./models/`.

| Gesture | Filename |
|---|---|
| Sonic | `sonic.gif` |
| Smirk | `smirk-meme.jpg` |
| Wink | `monkey-wink.jpg` |
| Speed | `speed.gif` |
| Patrick | `patrick-meme.jpg` |
| Shock | `shock-guy-meme.jpg` |
| Cut It Out | `cut-it.gif` |
| Shush | `dog-shush.jpg` |
| Thinking | `monkey-thinking.jpg` |
| LeBron | `lebron-scream.jpg` |
| Giggle | `baby-meme-giggle.gif` |
| Shaq T | `shaq.jpg` |
| Self Pointing | `jerry.gif` |
| Jerry | `jerry-meme.jpg` |
| Idle | `idle.jpg` |

> Images and models are excluded from the repo via `.gitignore`.
> You must supply your own meme files.

---

## Project Structure

```
AI-Meme/
├── images/          # Meme assets — not tracked by git
├── models/          # MediaPipe models — not tracked by git, auto-downloaded
├── main.py          # All app logic
├── requirements.txt
├── context.md
└── README.md
```

---

## Contributing

1. Fork the repo
2. Create a branch: `feature/your-gesture-name`
3. Add your gesture detection function and meme mapping
4. Open a PR with a description of the gesture and detection logic

---

## License

MIT
