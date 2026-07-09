# 🎧 Audio Transcriber

**Audio Transcriber** is a Python script for automatic speech-to-text transcription using [OpenAI Whisper](https://github.com/openai/whisper). Supports batch processing, SRT/TXT export, and flexible configuration via `settings.ini`.

---

## 📦 Description

- 🔊 Transcribe audio recordings (including speech) to text.
- 📂 Process all files in a specified folder (including subdirectories).
- 📝 Export results in:
  - `.txt` — with timestamps (`_timecodes.txt`)
  - `.txt` — plain text (`_raw.txt`)
  - `.srt` — subtitles (optional)
- ⚙️ Flexible configuration via `settings.ini`.
- 🧠 GPU acceleration (CUDA).
- 📊 Processing time and speed statistics.

---

## 🧰 Dependencies

```bash
pip install torch torchaudio
pip install openai-whisper
pip install pydub
pip install 'numpy<2.5'
```

> ⚠️ Numba (Whisper dependency) requires NumPy < 2.5. If you get `Numba needs NumPy 2.4 or less` after a system update — downgrade NumPy: `pip install 'numpy<2.5' --break-system-packages`.

> ⚠️ For CUDA support, install a CUDA-compatible version of `torch`.

### Arch Linux

```bash
sudo pacman -S python python-pip python-pydub python-openai-whisper
sudo pacman -Syu nvidia nvidia-utils cuda nvidia-open-dkms
# reboot
sudo pacman -S python-pytorch-cuda
```

### Ubuntu / Debian

```bash
sudo apt update
sudo apt install python3 python3-pip python3-pydub ffmpeg
pip install torch torchaudio openai-whisper 'numpy<2.5'
```

> For CUDA, install `python3-pytorch-cuda` via pip (see [pytorch.org](https://pytorch.org)).

### CentOS / RHEL / Fedora

```bash
sudo dnf install python3 python3-pip ffmpeg
pip install torch torchaudio openai-whisper pydub 'numpy<2.5'
```

> On CentOS/RHEL ffmpeg may require [EPEL](https://docs.fedoraproject.org/en-US/epel/): `sudo dnf install epel-release`.

### Windows

1. Install [Python 3.8+](https://www.python.org/downloads/) — check «Add Python to PATH» during setup.
2. Install [FFmpeg](https://ffmpeg.org/download.html) and add `bin\ffmpeg.exe` to `PATH`.
3. Open Command Prompt (`cmd`) and run:

```cmd
pip install torch torchaudio openai-whisper pydub 'numpy<2.5'
```

---

## 📁 Project Structure

```
speech_to_text.py          # Main script
settings.ini               # Configuration file
settings_badwords.ini      # Badword patterns file
models/                    # Whisper model directory (auto-created)
sources/                   # Audio files directory (default)
```

---

## ⚙️ Configuration (settings.ini)

```ini
[OPTIONS]
sources_dir = ./sources/
badwords_file = ./settings_badwords.ini
transcribe_engine = openai-whisper
whisper_model = base
force_transcribe_language = ru
model_path = ./models/
skip_transcoded_files = true
use_cuda = true
export_srt_file = true
export_raw_file = true
logging = true

[TRANSCRIBE]
beam_size = 5
temperature = 0.0,0.2,0.4,0.6,0.8,1.0
condition_on_prev_tokens = true
initial_prompt = 
compression_ratio_threshold = 2.4
logprob_threshold = -1.0
no_speech_threshold = 0.6
patience = 1.0
length_penalty = 1.0
suppress_blank = true
suppress_tokens = -1
without_timestamps = false
max_initial_timestamp = 1.0
fp16 = true
temperature_increment_on_fallback = 0.2
```

### 🔧 [OPTIONS] parameters:

| Parameter                   | Description |
|-----------------------------|-------------|
| `sources_dir`               | Path to audio files folder |
| `badwords_file`             | Path to badword patterns file (regex) |
| `transcribe_engine`         | Transcription engine (`openai-whisper`) |
| `whisper_model`             | Model: `tiny`, `tiny.en`, `base`, `base.en`, `small`, `small.en`, `medium`, `medium.en`, `large`, `large-v1`, `large-v2`, `large-v3`, `turbo` |
| `force_transcribe_language` | Language (`ru`, `en`, or empty for auto-detect) |
| `model_path`                | Path to models directory |
| `skip_transcoded_files`     | Skip already processed files |
| `use_cuda`                  | Use GPU (if available) |
| `export_srt_file`           | Export SRT subtitles |
| `export_raw_file`           | Export plain text |
| `logging`                   | Log sessions to `transcription.log` |

### 🧠 [TRANSCRIBE] parameters:

See [Whisper API documentation](https://github.com/openai/whisper/blob/main/whisper/transcribe.py#L391)

---

## 🚫 Badwords File (settings_badwords.ini)

Each non-empty line is a regex pattern. Lines starting with `#` are comments.

Example `settings_badwords.ini`:

```ini
# Advertisement inserts
Субтитры создал
Субтитры сделал
Субтитры от

# Noise markers
\(applause\)
\(laughter\)
(music)
```

> Patterns are applied via `re.sub()` with `IGNORECASE` flag. Escape literal dots with `\.`.

---

## ▶️ Run

```bash
python3 speech_to_text.py
```

---

## 📁 Input Formats

`.mp3`, `.wav`, `.flac`, `.aac`, `.ogg`, `.opus`, `.m4a`, `.wma`, `.aiff`, `.amr`

Scans the specified folder and all subdirectories.

---

## 📤 Output Files

Each audio file generates (in the same folder):

| File                          | Description |
|-------------------------------|-------------|
| `_timecodes.txt`              | Text with timestamps |
| `_raw.txt`                    | Plain text split into paragraphs |
| `.srt`                        | Subtitles (if enabled) |
| `_ERROR.txt`                  | Error log (if processing failed) |

---

## 📊 Example Output

```
==================================================
Audio Transcriber v1.0
Based on OpenAI Whisper
==================================================

🚀 CUDA enabled: 1 GPU(s) available
   GPU: NVIDIA GeForce RTX 3070
   CUDA version: 12.1

Scanning "sources/" (including subdirectories)...
✅ Found 3 audio file(s) to process.

Loading model: "base" using engine: openai-whisper...
✅ Model loaded.

[  1/3] Processing: example.mp3
    Duration: 00:02:15
    Starting transcription with openai-whisper (model: base)...
    Transcribing... (duration: 00:02:15)
✅ Done in 0:00:12

📊 Total statistics:
  🕐 Audio duration: 00:06:42
  ⏱️ Processing time: 00:00:35
  ⚡ Speed ratio: 11.54x
```

---

## 🧪 Requirements

- Python 3.8+
- CUDA (optional, for GPU acceleration)
- OS: Windows, Linux, macOS

---

## 🧾 License

MIT

