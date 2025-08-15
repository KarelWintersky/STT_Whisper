import os
import re
from datetime import datetime
from pydub import AudioSegment
import torch

def _parse_float(value, default=None):
    """Парсит float из строки, возвращает default если не удается."""
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        print(f"Предупреждение: Не удалось преобразовать '{value}' в float. Используется значение по умолчанию.")
        return default

def _parse_int(value, default=None):
    """Парсит int из строки, возвращает default если не удается."""
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        print(f"Предупреждение: Не удалось преобразовать '{value}' в int. Используется значение по умолчанию.")
        return default

def _parse_bool(value, default=None):
    """Парсит bool из строки ('true', 'false'), возвращает default если не удается."""
    if not value:
        return default
    value_lower = value.lower().strip()
    if value_lower in ('true', '1', 'yes', 'on'):
        return True
    elif value_lower in ('false', '0', 'no', 'off'):
        return False
    else:
        print(f"Предупреждение: Не удалось преобразовать '{value}' в bool. Используется значение по умолчанию.")
        return default

def _parse_list_of_floats(value, default=None):
    """Парсит список float из строки, разделенной запятыми."""
    if not value:
        return default
    try:
        return [float(item.strip()) for item in value.split(',') if item.strip()]
    except ValueError:
        print(f"Предупреждение: Не удалось преобразовать '{value}' в список float. Используется значение по умолчанию.")
        return default

def _parse_list_of_ints(value, default=None):
    """Парсит список int из строки, разделенной запятыми."""
    if not value:
        return default
    try:
        return [int(item.strip()) for item in value.split(',') if item.strip()]
    except ValueError:
        print(f"Предупреждение: Не удалось преобразовать '{value}' в список int. Используется значение по умолчанию.")
        return default

def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def print_progress_bar(current, total, bar_length=30):
    if total <= 0:
        return
    fraction = current / total
    filled = int(bar_length * fraction)
    bar = '█' * filled + '░' * (bar_length - filled)
    print(f'\r    Progress: |{bar}| {fraction:.1%} ({format_time(current)}/{format_time(total)})', end='', flush=True)

def match_ext(filename, extensions):
    ext = filename.lower().split('.')[-1]
    return ext in extensions

def process_audiofile(model, audio_path, file_index, total_files, USING_FASTER, audio_exts,
                     text_language, beam_size, temperature, condition_on_prev_tokens,
                     initial_prompt, compression_ratio_threshold, logprob_threshold,
                     no_speech_threshold, patience, length_penalty, suppress_blank,
                     suppress_tokens, without_timestamps, max_initial_timestamp,
                     fp16, temperature_increment_on_fallback, whisper_model):
    # Проверяем, существует ли файл
    if not os.path.exists(audio_path):
        print(f'[{file_index:3d}/{total_files}] ❌ File not found: {audio_path}')
        return

    file_start_time = datetime.now()
    print(f'[{file_index:3d}/{total_files}] Processing: {audio_path}')

    # Определяем длительность
    try:
        audio = AudioSegment.from_file(audio_path)
        duration = len(audio) / 1000.0
        print(f'    Duration: {format_time(duration)}')
    except Exception as e:
        print(f'    ⚠️ Could not read duration: {e}')
        duration = 0

    # Пути к выходным файлам
    dirname = os.path.dirname(audio_path)
    basename = os.path.basename(audio_path)
    name_noext = os.path.splitext(basename)[0]

    timecode_file = os.path.join(dirname, name_noext + '_timecodes.txt')
    rawtext_file = os.path.join(dirname, name_noext + '_raw.txt')

    try:
        if USING_FASTER:
            # --- Подготовка параметров для faster-whisper ---
            transcribe_kwargs = {
                "beam_size": beam_size,
                "language": text_language,
                "initial_prompt": initial_prompt,
                "temperature": temperature,
                "compression_ratio_threshold": compression_ratio_threshold,
                "log_prob_threshold": logprob_threshold,
                "no_speech_threshold": no_speech_threshold,
                "condition_on_prev_tokens": condition_on_prev_tokens,
                "patience": patience,
                "length_penalty": length_penalty,
                "suppress_blank": suppress_blank,
                "suppress_tokens": suppress_tokens,
                "without_timestamps": without_timestamps,
                "max_initial_timestamp": max_initial_timestamp,
            }
            # Удаляем ключи со значением None
            transcribe_kwargs = {k: v for k, v in transcribe_kwargs.items() if v is not None}
            # --- Конец подготовки параметров ---

            print(f"    Starting transcription with faster-whisper (model: {whisper_model})...")
            segments, info = model.transcribe(audio_path, **transcribe_kwargs)
            print(f"    Detected language: {info.language} (prob: {info.language_probability:.2f})")
            if duration > 0:
                print_progress_bar(0, duration)
            else:
                print("    Transcribing... (duration unknown)")

            full_text = []
            with open(timecode_file, 'w', encoding='UTF-8') as f:
                for segment in segments:
                    start = segment.start
                    end = segment.end
                    text = segment.text.strip()
                    hh, mm, ss = int(start // 3600), int((start % 3600) // 60), int(start % 60)
                    f.write(f"[{hh:02d}:{mm:02d}:{ss:02d}] {text}\n")
                    full_text.append(text)
                    if duration > 0:
                        print_progress_bar(end, duration)

            if duration > 0:
                print_progress_bar(duration, duration)
            print()

        else:  # openai-whisper
            # --- Подготовка параметров для openai-whisper ---
            transcribe_kwargs = {
                "verbose": False,
                "language": text_language,
                "temperature": temperature,
                "condition_on_previous_text": condition_on_prev_tokens,
                "initial_prompt": initial_prompt,
                "compression_ratio_threshold": compression_ratio_threshold,
                "logprob_threshold": logprob_threshold,
                "no_speech_threshold": no_speech_threshold,
                "patience": patience,
                "length_penalty": length_penalty,
                "suppress_blank": suppress_blank,
                "suppress_tokens": suppress_tokens,
                "without_timestamps": without_timestamps,
                "max_initial_timestamp": max_initial_timestamp,
                "fp16": fp16,
            }
            # Если beam_size задано, передаем его
            if beam_size is not None:
                transcribe_kwargs["beam_size"] = beam_size
            # Удаляем ключи со значением None
            transcribe_kwargs = {k: v for k, v in transcribe_kwargs.items() if v is not None}
            # --- Конец подготовки параметров ---

            print(f"    Starting transcription with openai-whisper (model: {whisper_model})...")
            result = model.transcribe(audio_path, **transcribe_kwargs)
            # Оценка длительности (если не получилось через pydub)
            if duration == 0 and 'segments' in result and len(result['segments']) > 0:
                duration = result['segments'][-1]['end']
                print("    Transcribing... (duration estimated)")
            else:
                print(f"    Transcribing... (duration: {format_time(duration)})")

            full_text = []
            with open(timecode_file, 'w', encoding='UTF-8') as f:
                for i, segment in enumerate(result['segments']):
                    start = segment['start']
                    end = segment['end']
                    text = segment['text'].strip()
                    hh, mm, ss = int(start // 3600), int((start % 3600) // 60), int(start % 60)
                    f.write(f"[{hh:02d}:{mm:02d}:{ss:02d}] {text}\n")
                    full_text.append(text)
                    if duration > 0:
                        print_progress_bar(end, duration)

            if duration > 0:
                print_progress_bar(duration, duration)
            print()

        # Сохраняем сырой текст
        rawtext = ' '.join(full_text)
        rawtext = re.sub(r" +", " ", rawtext)
        # Разбиваем на абзацы по знакам препинания
        alltext = re.sub(r"([.!?])\s+", r"\1\n", rawtext)
        # Удаляем возможные пустые строки в начале/конце
        alltext = alltext.strip()

        with open(rawtext_file, 'w', encoding='UTF-8') as f:
            f.write(alltext)

        print(f'✅ Done in {datetime.now() - file_start_time}')

    except Exception as e:
        print(f'\n❌ Error processing {audio_path}: {e}')
        # Записываем ошибку в файл
        error_file_path = os.path.join(dirname, name_noext + '_ERROR.txt')
        with open(error_file_path, 'w', encoding='UTF-8') as ef:
            ef.write(f"Error processing {audio_path}: {e}\n")


def load_model(whisper_model, engine_name, USING_FASTER):
    """Загрузка модели транскрипции"""
    print(f'Loading model: "{whisper_model}" using engine: {engine_name}...')
    if USING_FASTER:
        model = WhisperModel(
            whisper_model,
            device="cuda" if torch.cuda.is_available() else "cpu",
            compute_type="float16" if torch.cuda.is_available() else "int8",
            download_root="./models"
        )
    else:
        model = whisper.load_model(
            whisper_model,
            device="cuda" if torch.cuda.is_available() else "cpu",
            download_root="./models/openai-whisper"
        )
    print('✅ Model loaded.\n')
    return model

def process_audiofile(model, audio_path, file_index, total_files, USING_FASTER, audio_exts,
                     text_language, beam_size, temperature, condition_on_prev_tokens,
                     initial_prompt, compression_ratio_threshold, logprob_threshold,
                     no_speech_threshold, patience, length_penalty, suppress_blank,
                     suppress_tokens, without_timestamps, max_initial_timestamp,
                     fp16, temperature_increment_on_fallback, whisper_model):
    """Основная функция обработки аудиофайла - вызывает соответствующую реализацию"""
    if USING_FASTER:
        process_audiofile_fasterwhisper(
            model, audio_path, file_index, total_files,
            text_language, beam_size, temperature, condition_on_prev_tokens,
            initial_prompt, compression_ratio_threshold, logprob_threshold,
            no_speech_threshold, patience, length_penalty, suppress_blank,
            suppress_tokens, without_timestamps, max_initial_timestamp,
            whisper_model
        )
    else:
        process_audiofile_openai_whisper(
            model, audio_path, file_index, total_files,
            text_language, beam_size, temperature, condition_on_prev_tokens,
            initial_prompt, compression_ratio_threshold, logprob_threshold,
            no_speech_threshold, patience, length_penalty, suppress_blank,
            suppress_tokens, without_timestamps, max_initial_timestamp,
            fp16, whisper_model
        )


def process_audiofile_openai_whisper(model, audio_path, file_index, total_files,
                                    text_language, beam_size, temperature, condition_on_prev_tokens,
                                    initial_prompt, compression_ratio_threshold, logprob_threshold,
                                    no_speech_threshold, patience, length_penalty, suppress_blank,
                                    suppress_tokens, without_timestamps, max_initial_timestamp,
                                    fp16, whisper_model):
    """Обработка аудиофайла с использованием openai-whisper"""
    # Проверяем, существует ли файл
    if not os.path.exists(audio_path):
        print(f'[{file_index:3d}/{total_files}] ❌ File not found: {audio_path}')
        return

    file_start_time = datetime.now()
    print(f'[{file_index:3d}/{total_files}] Processing: {audio_path}')

    # Определяем длительность
    try:
        audio = AudioSegment.from_file(audio_path)
        duration = len(audio) / 1000.0
        print(f'    Duration: {format_time(duration)}')
    except Exception as e:
        print(f'    ⚠️ Could not read duration: {e}')
        duration = 0

    # Пути к выходным файлам
    dirname = os.path.dirname(audio_path)
    basename = os.path.basename(audio_path)
    name_noext = os.path.splitext(basename)[0]

    timecode_file = os.path.join(dirname, name_noext + '_timecodes.txt')
    rawtext_file = os.path.join(dirname, name_noext + '_raw.txt')

    try:
        # --- Подготовка параметров для openai-whisper ---
        transcribe_kwargs = {
            "verbose": False,
            "language": text_language,
            "temperature": temperature,
            "condition_on_previous_text": condition_on_prev_tokens,
            "initial_prompt": initial_prompt,
            "compression_ratio_threshold": compression_ratio_threshold,
            "logprob_threshold": logprob_threshold,
            "no_speech_threshold": no_speech_threshold,
            "patience": patience,
            "length_penalty": length_penalty,
            "suppress_blank": suppress_blank,
            "suppress_tokens": suppress_tokens,
            "without_timestamps": without_timestamps,
            "max_initial_timestamp": max_initial_timestamp,
            "fp16": fp16,
        }

        # Если beam_size задано, передаем его
        if beam_size is not None:
            transcribe_kwargs["beam_size"] = beam_size

        # Удаляем ключи со значением None
        transcribe_kwargs = {k: v for k, v in transcribe_kwargs.items() if v is not None}
        # --- Конец подготовки параметров ---

        print(f"    Starting transcription with openai-whisper (model: {whisper_model})...")
        result = model.transcribe(audio_path, **transcribe_kwargs)

        # Оценка длительности (если не получилось через pydub)
        if duration == 0 and 'segments' in result and len(result['segments']) > 0:
            duration = result['segments'][-1]['end']
            print("    Transcribing... (duration estimated)")
        else:
            print(f"    Transcribing... (duration: {format_time(duration)})")

        full_text = []
        with open(timecode_file, 'w', encoding='UTF-8') as f:
            for i, segment in enumerate(result['segments']):
                start = segment['start']
                end = segment['end']
                text = segment['text'].strip()
                hh, mm, ss = int(start // 3600), int((start % 3600) // 60), int(start % 60)
                f.write(f"[{hh:02d}:{mm:02d}:{ss:02d}] {text}\n")
                full_text.append(text)
                if duration > 0:
                    print_progress_bar(end, duration)

        if duration > 0:
            print_progress_bar(duration, duration)
        print()

        # Сохраняем сырой текст
        rawtext = ' '.join(full_text)
        rawtext = re.sub(r" +", " ", rawtext)

        # Разбиваем на абзацы по знакам препинания
        alltext = re.sub(r"([.!?])\s+", r"\1\n", rawtext)

        # Удаляем возможные пустые строки в начале/конце
        alltext = alltext.strip()

        with open(rawtext_file, 'w', encoding='UTF-8') as f:
            f.write(alltext)

        print(f'✅ Done in {datetime.now() - file_start_time}')

    except Exception as e:
        print(f'\n❌ Error processing {audio_path}: {e}')
        # Записываем ошибку в файл
        error_file_path = os.path.join(dirname, name_noext + '_ERROR.txt')
        with open(error_file_path, 'w', encoding='UTF-8') as ef:
            ef.write(f"Error processing {audio_path}: {e}\n")


def process_audiofile_fasterwhisper(model, audio_path, file_index, total_files, 
                                   text_language, beam_size, temperature, condition_on_prev_tokens,
                                   initial_prompt, compression_ratio_threshold, logprob_threshold,
                                   no_speech_threshold, patience, length_penalty, suppress_blank,
                                   suppress_tokens, without_timestamps, max_initial_timestamp,
                                   whisper_model):
    """Обработка аудиофайла с использованием faster-whisper"""
    # Проверяем, существует ли файл
    if not os.path.exists(audio_path):
        print(f'[{file_index:3d}/{total_files}] ❌ File not found: {audio_path}')
        return

    file_start_time = datetime.now()
    print(f'[{file_index:3d}/{total_files}] Processing: {audio_path}')

    # Определяем длительность
    try:
        audio = AudioSegment.from_file(audio_path)
        duration = len(audio) / 1000.0
        print(f'    Duration: {format_time(duration)}')
    except Exception as e:
        print(f'    ⚠️ Could not read duration: {e}')
        duration = 0

    # Пути к выходным файлам
    dirname = os.path.dirname(audio_path)
    basename = os.path.basename(audio_path)
    name_noext = os.path.splitext(basename)[0]

    timecode_file = os.path.join(dirname, name_noext + '_timecodes.txt')
    rawtext_file = os.path.join(dirname, name_noext + '_raw.txt')

    try:
        # --- Подготовка параметров для faster-whisper ---
        transcribe_kwargs = {
            "beam_size": beam_size,
            "language": text_language,
            "initial_prompt": initial_prompt,
            "temperature": temperature,
            "compression_ratio_threshold": compression_ratio_threshold,
            "log_prob_threshold": logprob_threshold,
            "no_speech_threshold": no_speech_threshold,
            "condition_on_prev_tokens": condition_on_prev_tokens,
            "patience": patience,
            "length_penalty": length_penalty,
            "suppress_blank": suppress_blank,
            "suppress_tokens": suppress_tokens,
            "without_timestamps": without_timestamps,
            "max_initial_timestamp": max_initial_timestamp,
        }
        # Удаляем ключи со значением None
        transcribe_kwargs = {k: v for k, v in transcribe_kwargs.items() if v is not None}
        # --- Конец подготовки параметров ---

        print(f"    Starting transcription with faster-whisper (model: {whisper_model})...")
        segments, info = model.transcribe(audio_path, **transcribe_kwargs)
        print(f"    Detected language: {info.language} (prob: {info.language_probability:.2f})")
        if duration > 0:
            print_progress_bar(0, duration)
        else:
            print("    Transcribing... (duration unknown)")

        full_text = []
        with open(timecode_file, 'w', encoding='UTF-8') as f:
            for segment in segments:
                start = segment.start
                end = segment.end
                text = segment.text.strip()
                hh, mm, ss = int(start // 3600), int((start % 3600) // 60), int(start % 60)
                f.write(f"[{hh:02d}:{mm:02d}:{ss:02d}] {text}\n")
                full_text.append(text)
                if duration > 0:
                    print_progress_bar(end, duration)

        if duration > 0:
            print_progress_bar(duration, duration)
        print()

        # Сохраняем сырой текст
        rawtext = ' '.join(full_text)
        rawtext = re.sub(r" +", " ", rawtext)
        # Разбиваем на абзацы по знакам препинания
        alltext = re.sub(r"([.!?])\s+", r"\1\n", rawtext)
        # Удаляем возможные пустые строки в начале/конце
        alltext = alltext.strip()

        with open(rawtext_file, 'w', encoding='UTF-8') as f:
            f.write(alltext)

        print(f'✅ Done in {datetime.now() - file_start_time}')

    except Exception as e:
        print(f'\n❌ Error processing {audio_path}: {e}')
        # Записываем ошибку в файл
        error_file_path = os.path.join(dirname, name_noext + '_ERROR.txt')
        with open(error_file_path, 'w', encoding='UTF-8') as ef:
            ef.write(f"Error processing {audio_path}: {e}\n")


def initialize_engine(engine_name, whisper_model):
    """Инициализация движка транскрипции и загрузка модели"""
    # Выбор движка транскрибации
    if engine_name == "faster-whisper":
        from faster_whisper import WhisperModel
        USING_FASTER = True
        print(f'Loading model: "{whisper_model}" using engine: {engine_name}...')
        model = WhisperModel(
            whisper_model,
            device="cuda" if torch.cuda.is_available() else "cpu",
            compute_type="float16" if torch.cuda.is_available() else "int8",
            download_root="./models"
        )
    elif engine_name == "openai-whisper":
        import whisper
        USING_FASTER = False
        print(f'Loading model: "{whisper_model}" using engine: {engine_name}...')
        model = whisper.load_model(
            whisper_model,
            device="cuda" if torch.cuda.is_available() else "cpu",
            download_root="./models/openai-whisper"
        )
    else:
        raise ValueError(f"Unknown transcribe_engine: '{engine_name}'. Use 'faster-whisper' or 'openai-whisper'.")
    
    print('✅ Model loaded.\n')
    return USING_FASTER, model

