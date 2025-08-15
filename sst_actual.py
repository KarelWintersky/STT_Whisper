#!/usr/bin/python3
# FasterWhisper model not recommended
# pip install pydub configparser openai-whisper
# pip install faster-whisper

import os
import re
import configparser
from datetime import datetime
from pydub import AudioSegment
import torch

# Импортируем функции из библиотеки
from audio_transcriber_lib import (
	_parse_float, _parse_int, _parse_bool, _parse_list_of_floats, _parse_list_of_ints,
	format_time, print_progress_bar, match_ext, 
	initialize_engine, load_model, process_audiofile
)

# Читаем настройки
config = configparser.ConfigParser()
config.read("settings.ini")

# --- Параметры из конфигурации ---
try:
    audio_folder = config["OPTIONS"]["sources_dir"]
    engine_name = config["OPTIONS"]["transcribe_engine"].strip()
    whisper_model = config["OPTIONS"]["whisper_model"]
    text_language = config["OPTIONS"].get("force_transcribe_language", "").strip()
    text_language = text_language if text_language else None
except KeyError as e:
    print(f"Ошибка: Отсутствует обязательный параметр в settings.ini: {e}")
    exit(1)

# Загружаем и парсим параметры декодирования
beam_size					= _parse_int(config["TRANSCRIBE"].get("beam_size"), default=None)
temperature_raw 			= config["TRANSCRIBE"].get("temperature", "").strip()
temperature 				= _parse_list_of_floats(temperature_raw, default=None) if ',' in temperature_raw else _parse_float(temperature_raw, default=None)
condition_on_prev_tokens	= _parse_bool(config["TRANSCRIBE"].get("condition_on_prev_tokens"), default=None)
initial_prompt 				= config["TRANSCRIBE"].get("initial_prompt", "").strip() or None
compression_ratio_threshold = _parse_float(config["TRANSCRIBE"].get("compression_ratio_threshold"), default=None)
logprob_threshold			= _parse_float(config["TRANSCRIBE"].get("logprob_threshold"), default=None)
no_speech_threshold 		= _parse_float(config["TRANSCRIBE"].get("no_speech_threshold"), default=None)
patience 					= _parse_float(config["TRANSCRIBE"].get("patience"), default=None)
length_penalty 				= _parse_float(config["TRANSCRIBE"].get("length_penalty"), default=None)
suppress_blank 				= _parse_bool(config["TRANSCRIBE"].get("suppress_blank"), default=None)
suppress_tokens_raw 		= config["TRANSCRIBE"].get("suppress_tokens", "").strip()
suppress_tokens 			= _parse_list_of_ints(suppress_tokens_raw, default=None)
without_timestamps 			= _parse_bool(config["TRANSCRIBE"].get("without_timestamps"), default=None)
max_initial_timestamp 		= _parse_float(config["TRANSCRIBE"].get("max_initial_timestamp"), default=None)
fp16						= _parse_bool(config["TRANSCRIBE"].get("fp16"), default=None) # Только для openai-whisper
temperature_increment_on_fallback = _parse_float(config["TRANSCRIBE"].get("temperature_increment_on_fallback"), default=None) # Только для openai-whisper

# Поддерживаемые расширения
audio_exts = ['mp3', 'aac', 'ogg', 'wav', 'opus', 'flac', 'm4a', 'wma', 'aiff', 'amr']

def main():
    start_time = datetime.now()
    print(f'Scanning "{audio_folder}" (including subdirectories)...')

    if not os.path.exists(audio_folder):
        print(f'❌ Directory not found: "{audio_folder}"')
        return

    # Поиск аудиофайлов
    audio_files = []
    for root, dirs, files in os.walk(audio_folder):
        for file in files:
            if match_ext(file, audio_exts):
                full_path = os.path.join(root, file)
                audio_files.append(full_path)

    total_files = len(audio_files)
    if total_files == 0:
        print('No audio files found to process.')
        return

    print(f'Found {total_files} audio file(s) to process.\n')

    # Инициализируем движок и загружаем модель только если есть файлы для обработки
    USING_FASTER, model = initialize_engine(engine_name, whisper_model)

    # Обработка
    for idx, audio_file in enumerate(audio_files, 1):
        process_audiofile(
            model, audio_file, idx, total_files, USING_FASTER, audio_exts,
            text_language, beam_size, temperature, condition_on_prev_tokens,
            initial_prompt, compression_ratio_threshold, logprob_threshold,
            no_speech_threshold, patience, length_penalty, suppress_blank,
            suppress_tokens, without_timestamps, max_initial_timestamp,
            fp16, temperature_increment_on_fallback, whisper_model
        )

    print('✅ All files processed.')
    print(f'Total time: {datetime.now() - start_time}')



if __name__ == '__main__':
    main()

