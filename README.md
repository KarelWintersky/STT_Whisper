# 🎧 Audio Transcriber

**Audio Transcriber** — это Python-скрипт для автоматической транскрипции аудиофайлов в текст с использованием модели [OpenAI Whisper](https://github.com/openai/whisper). Поддерживает пакетную обработку файлов, экспорт в форматы SRT и TXT, а также гибкую настройку через конфигурационный файл.

---

## 📦 Описание

Этот скрипт предназначен для:

- 🔊 Транскрипции аудиозаписей (в том числе речи) в текст.
- 📂 Обработки всех файлов в указанной папке (включая подкаталоги).
- 📝 Экспорта результатов в форматы:
  - `.txt` — с временными метками (`_timecodes.txt`)
  - `.txt` — чистый текст (`_raw.txt`)
  - `.srt` — субтитры (опционально)
- ⚙️ Гибкой настройки через `settings.ini`.
- 🧠 Поддержки GPU (через CUDA) для ускорения обработки.
- 📊 Вывода статистики по времени обработки и скорости.

---

## 🧰 Установка зависимостей

Перед запуском скрипта установите необходимые библиотеки:

```bash
pip install torch torchaudio
pip install openai-whisper
pip install pydub
pip install 'numpy<2.5'
```

> ⚠️ Numba (зависимость Whisper) требует NumPy < 2.5. Если после обновления системы получили ошибку `Numba needs NumPy 2.4 or less` — откатите NumPy: `pip install 'numpy<2.5' --break-system-packages`.

> ⚠️ Для работы с CUDA убедитесь, что у вас установлена совместимая версия `torch` с поддержкой CUDA.

### Ubuntu / Debian

```bash
sudo apt update
sudo apt install python3 python3-pip python3-pydub ffmpeg
pip install torch torchaudio openai-whisper 'numpy<2.5'
```

> Для CUDA установите torch с поддержкой CUDA через pip (см. [pytorch.org](https://pytorch.org)).

### CentOS / RHEL / Fedora

```bash
sudo dnf install python3 python3-pip ffmpeg
pip install torch torchaudio openai-whisper pydub 'numpy<2.5'
```

> На CentOS/RHEL ffmpeg может потребовать [EPEL](https://docs.fedoraproject.org/en-US/epel/): `sudo dnf install epel-release`.

### Arch Linux

```bash
sudo pacman -S python python-pip python-pydub python-openai-whisper
sudo pacman -Syu nvidia nvidia-utils cuda nvidia-open-dkms
# перезагрузка
sudo pacman -S python-pytorch-cuda
```

### Windows

1. Установите [Python 3.8+](https://www.python.org/downloads/) — при установке отметьте «Add Python to PATH».
2. Установите [FFmpeg](https://ffmpeg.org/download.html) и добавьте путь к `bin\ffmpeg.exe` в `PATH`.
3. Откройте командную строку (`cmd`) и выполните:

```cmd
pip install torch torchaudio openai-whisper pydub 'numpy<2.5'
```

---

## 📁 Структура проекта

```
speech_to_text.py          # Основной скрипт
settings.ini               # Конфигурационный файл
settings_badwords.ini      # Файл с паттернами нежелательных слов
models/                    # Папка для моделей Whisper (создаётся автоматически)
sources/                   # Папка с аудиофайлами (по умолчанию)
```

---

## ⚙️ Конфигурация (settings.ini)

Пример конфигурационного файла:

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
temperature = 0.0
condition_on_prev_tokens = false
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

### 🔧 Параметры [OPTIONS]:

| Параметр                     | Описание |
|-----------------------------|----------|
| `sources_dir`               | Путь к папке с аудиофайлами |
| `badwords_file`             | Путь к файлу с паттернами нежелательных слов (regex) |
| `transcribe_engine`         | Движок транскрипции (`openai-whisper`) |
| `whisper_model`             | Модель Whisper: `tiny`, `tiny.en`, `base`, `base.en`, `small`, `small.en`, `medium`, `medium.en`, `large`, `large-v1`, `large-v2`, `large-v3`, `turbo` |
| `force_transcribe_language` | Язык транскрипции (например, `ru`, `en`). Оставьте пустым для автоопределения |
| `model_path`                | Путь к папке с моделями |
| `skip_transcoded_files`     | Пропускать уже обработанные файлы |
| `use_cuda`                  | Использовать GPU (если доступен) |
| `export_srt_file`           | Экспортировать SRT-субтитры |
| `export_raw_file`           | Экспортировать чистый текст |
| `logging`                   | Логировать сессии в `transcription.log` |

### 🧠 Параметры [TRANSCRIBE]:

См. документацию [Whisper API](https://github.com/openai/whisper/blob/main/whisper/transcribe.py#L391)

#### ⚠️ Предотвращение зацикливания

Некоторые комбинации параметров могут вызывать зацикливание (бесконечную генерацию). Вот безопасные настройки:

| Параметр | Безопасно | Опасно | Пояснение |
|----------|-----------|--------|-----------|
| `temperature` | `0.0` (одно число) | `0.0,0.2,0.4,...` (список) | Список температур включает fallback-режим, который на длинных файлах может зациклить Whisper |
| `condition_on_prev_tokens` | `false` | `true` | Учёт предыдущего текста полезен для связности, но на длинных файлах часто вызывает зацикливание |
| `initial_prompt` | пустая строка | любой текст | Промпт может усугубить зацикливание, особенно если он длинный |

**Рекомендуется для длинных аудиофайлов (>30 мин):**
```ini
temperature = 0.0
condition_on_prev_tokens = false
initial_prompt =
```

---

## 🚫 Файл плохих слов (settings_badwords.ini)

Файл содержит список паттернов (regex) для удаления нежелательных фраз из расшифровки. Каждая непустая строка — отдельный regex. Строки, начинающиеся с `#`, игнорируются.

Пример `settings_badwords.ini`:

```ini
# Рекламные вставки
Субтитры создал
Субтитры сделал
Субтитры от

# Шумовые пометки
\(аплодисменты\)
\(смех\)
(музыка)
```

> Паттерны применяются через `re.sub()` с флагом `IGNORECASE`. Если в regex нужна точка как литерал — экранируйте её (`\.`).

---

## ▶️ Запуск

```bash
python3 speech_to_text.py
```

---

## 📁 Входные данные

Поддерживаются следующие форматы аудио:

- `.mp3`, `.wav`, `.flac`, `.aac`, `.ogg`, `.opus`, `.m4a`, `.wma`, `.aiff`, `.amr`

Скрипт автоматически сканирует указанную папку и все вложенные каталоги.

---

## 📤 Выходные данные

Для каждого файла создаются следующие файлы (в той же папке):

| Файл                          | Описание |
|-------------------------------|----------|
| `_timecodes.txt`              | Текст с временными метками |
| `_raw.txt`                    | Чистый текст с разбивкой на абзацы |
| `.srt`                        | Субтитры (если включено) |
| `_ERROR.txt`                  | Лог ошибок (если произошла ошибка) |

---

## 📊 Пример вывода

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

## 🧪 Требования

- Python 3.8+
- CUDA (опционально, для GPU-ускорения)
- Поддерживаемая ОС: Windows, Linux, macOS

---

## 🧾 Лицензия

MIT

---

## 📬 Автор

Разработано для автоматической обработки аудиозаписей. Основано на [OpenAI Whisper](https://github.com/openai/whisper).

--- 

Если у вас есть вопросы или предложения — создавайте issue в репозитории.