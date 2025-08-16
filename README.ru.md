## Пакетная расшифровка речи в текст

**Вход**: аудиофайл

**Выход**: текст + текст с таймкодами

### Зависимости

- [Python](https://python.org) 3.10+
- Библиотека [Whisper](https://github.com/openai/whisper)
- Скомпилированный [FFMPEG](https://ffmpeg.org/download.html) + путь к нему в переменной окружения path
- Желательно GPU, но на CPU тоже работает (медленно)

### Подготовка

1. Установите упомянутые выше зависимости.
2. Задайте параметры распознавания в начале скрипта.
   - audio_folder - папка с аудиофайлами для распознавания в текст
   - whisper_model - размер модели, от которого зависит скоость и качество распознавания. Подробности о моделях есть в [readme](https://github.com/openai/whisper#available-models-and-languages) Whisper
   - text_language - код языка для распознавания. Поддерживаются все популярные языки, включая русский

### Использование

1. Запустите batch-speech-to-text.bat. Если все зависимости установлены, загрузится модель и начнется распознавание найденных в указанной папке аудиофайлов.
2. По ходу работы программы отображаются фрагменты распознанного текста.
3. По завершении работы программы рядом с исходным файлом появятся два текстовых файла:
   - filename.txt - распознанный текст, по одному предложению в строке.
   - filename_timecode.txt - распознанный текст с таймкодами, по ~3-5 секунд в строке.


# Простейший скрипт:

```python
#!/usr/bin/python3
# Install dependencies:
# pip install -U openai-whisper 
# OR
# pip install git+https://github.com/openai/whisper.git 
# OR, UPDATE
# pip install --upgrade --no-deps --force-reinstall git+https://github.com/openai/whisper.git 
# Install ffmpeg:
# sudo apt update && sudo apt install ffmpeg (debian)
# brew install ffmpeg (macOS)
# choco install ffmpeg (Windows 7+, https://chocolatey.org/)
# More info: https://github.com/openai/whisper

import os
import re
import whisper
from datetime import datetime
import time
import configparser

config = configparser.ConfigParser()
config.read("settings.ini")

# Load settings from config 
audio_folder = config["OPTIONS"]["sources_dir"]
whisper_model = config["OPTIONS"]["whisper_model"]
text_language = config["OPTIONS"]["force_transcribe_language"]

# Audio files with specified extensions will be processed
audio_exts = ['mp3', 'aac', 'ogg', 'wav', 'opus']

def main():
    start_time = datetime.now()

    print(f'Looking into "{audio_folder}"')
    os.chdir(audio_folder)

    files = [file for file in os.listdir(audio_folder) if match_ext(file, audio_exts)]
    print(f'Found {len(files)} files:')
    for filename in files: print(filename)

    for filename in files:
        print(f'\nProcessing {filename}')
        audio_file = os.path.join(audio_folder, filename)
        process_audiofile(audio_file)

    print('Total time processing: ', datetime.now() - start_time)
    
def match_ext(filename, extensions):
    return filename.split('.')[-1] in extensions

def process_audiofile(fname):
    file_start_time = datetime.now()
    
    fext = fname.split('.')[-1]
    fname_noext = fname[:-(len(fext)+1)]

    model = whisper.load_model(whisper_model)

    result = model.transcribe(fname, verbose = True, language = text_language)

    with open(fname_noext + '_timecodes.txt', 'w', encoding='UTF-8') as f:
        for segment in result['segments']:
            timecode_sec = int(segment['start'])
            hh = timecode_sec // 3600
            mm = (timecode_sec % 3600) // 60
            ss = timecode_sec % 60
            timecode = f'[{str(hh).zfill(2)}:{str(mm).zfill(2)}:{str(ss).zfill(2)}]'
            text = segment['text']
            f.write(f'{timecode} {text}\n')

    rawtext = ' '.join([segment['text'].strip() for segment in result['segments']])
    rawtext = re.sub(" +", " ", rawtext)

    alltext = re.sub("([\.\!\?]) ", "\\1\n", rawtext)

    with open(fname_noext + '_raw.txt', 'w', encoding='UTF-8') as f:
        f.write(alltext)

    print('Processing file ', fname, ' took ', datetime.now() - file_start_time, '\n')

# Calling main() function
if __name__ == '__main__':
    main()
```

