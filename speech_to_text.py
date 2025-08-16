#!/usr/bin/python3
# All-in-one version

import os
import re
import sys
import configparser
import torch
import warnings
import signal
import logging
from datetime import datetime
from pydub import AudioSegment
from pathlib import Path

# Подавляем предупреждение от whisper
warnings.filterwarnings("ignore", message="Performing inference on CPU when CUDA is available")

# ------------------------------------------------------------------------- #

class Helper:
    """Вспомогательный класс с утилитарными методами"""

    @staticmethod
    def format_time(seconds):
        """Форматирование времени"""
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    @staticmethod
    def format_elapsed_time(elapsed_time):
        """Форматирование времени выполнения без дробной части секунд"""
        return str(elapsed_time).split('.')[0]

    @staticmethod
    def format_srt_timestamp(seconds):
        """Форматирование времени для SRT (ЧЧ:ММ:СС,ммм)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        milliseconds = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"

    @staticmethod
    def match_ext(filename, extensions):
        """Проверка расширения файла"""
        ext = filename.lower().split('.')[-1]
        return ext in extensions

    @staticmethod
    def parse_float(value, default=None):
        """Парсит float из строки, возвращает default если не удается."""
        if not value:
            return default
        try:
            return float(value)
        except ValueError:
            print(f"Предупреждение: Не удалось преобразовать '{value}' в float. Используется значение по умолчанию.")
            return default

    @staticmethod
    def parse_int(value, default=None):
        """Парсит int из строки, возвращает default если не удается."""
        if not value:
            return default
        try:
            return int(value)
        except ValueError:
            print(f"Предупреждение: Не удалось преобразовать '{value}' в int. Используется значение по умолчанию.")
            return default

    @staticmethod
    def parse_bool(value, default=None):
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

    @staticmethod
    def parse_list_of_floats(value, default=None):
        """Парсит список float из строки, разделенной запятыми."""
        if not value:
            return default
        try:
            return [float(item.strip()) for item in value.split(',') if item.strip()]
        except ValueError:
            print(
                f"Предупреждение: Не удалось преобразовать '{value}' в список float. Используется значение по умолчанию.")
            return default

    @staticmethod
    def parse_list_of_ints(value, default=None):
        """Парсит список int из строки, разделенной запятыми."""
        if not value:
            return default
        try:
            return [int(item.strip()) for item in value.split(',') if item.strip()]
        except ValueError:
            print(
                f"Предупреждение: Не удалось преобразовать '{value}' в список int. Используется значение по умолчанию.")
            return default

    @staticmethod
    def print_total_stats(processed_files_count, total_duration, total_processing_time):
        """Вывод суммарной статистики"""
        if processed_files_count > 0:
            speed_ratio = total_duration / total_processing_time if total_processing_time > 0 else 0
            print(f"\n📊 Total statistics:")
            print(f"  🕐 Audio duration: {Helper.format_time(total_duration)}")
            print(f"  ⏱️ Processing time: {Helper.format_time(total_processing_time)}")
            print(f"  ⚡ Speed ratio: {speed_ratio:.2f}x")
        else:
            print("\nNo files were processed.")

    @staticmethod
    def print_progress_bar(current, total, bar_length=30):
        """Вывод прогресс-бара после кодирования файла (лишнего) """
        if total <= 0:
            return
        fraction = current / total
        filled = int(bar_length * fraction)
        bar = '█' * filled + '░' * (bar_length - filled)
        print(f'\r    Progress: |{bar}| {fraction:.1%} ({Helper.format_time(current)}/{Helper.format_time(total)})', end='', flush=True)

    @staticmethod
    def save_timecode_file(result, timecode_file):
        full_text = []

        with open(timecode_file, 'w', encoding='UTF-8') as f:
            for i, segment in enumerate(result['segments']):
                start = segment['start']
                end = segment['end']
                text = segment['text'].strip()
                hh, mm, ss = int(start // 3600), int((start % 3600) // 60), int(start % 60)
                f.write(f"[{hh:02d}:{mm:02d}:{ss:02d}] {text}\n")
                full_text.append(text)

        return full_text

    @staticmethod
    def save_text_files(full_text, rawtext_file):
        """Сохранение текстовых файлов"""

        # Сохраняем сырой текст
        rawtext = ' '.join(full_text)
        rawtext = re.sub(r" +", " ", rawtext)

        # Разбиваем на абзацы по знакам препинания
        alltext = re.sub(r"([.!?])\s+", r"\1\n", rawtext)

        # Удаляем возможные пустые строки в начале/конце
        alltext = alltext.strip()

        with open(rawtext_file, 'w', encoding='UTF-8') as f:
            f.write(alltext)

    @staticmethod
    def save_srt_file(result, srt_file_path):
        """Экспорт субтитров"""
        try:
            with open(srt_file_path, 'w', encoding='UTF-8') as f:
                for i, segment in enumerate(result['segments'], 1):
                    # Номер титра
                    f.write(f"{i}\n")

                    # Временные метки
                    start_time = Helper.format_srt_timestamp(segment['start'])
                    end_time = Helper.format_srt_timestamp(segment['end'])
                    f.write(f"{start_time} --> {end_time}\n")

                    # Текст субтитра
                    text = segment['text'].strip()
                    f.write(f"{text}\n")

                    # Пустая строка между субтитрами
                    f.write("\n")

            print(f"    SRT subtitles saved to: {os.path.basename(srt_file_path)}")

        except Exception as e:
            print(f"    ⚠️ Error exporting SRT: {e}")

# --------------------------------------------------------------------------------------------- #

class ConfigParser:
    """Класс для парсинга конфигурации"""
    
    def __init__(self, config_path="settings.ini"):
        self.config = configparser.ConfigParser()
        self.config.read(config_path)
        self._parse_config()
    
    def _parse_config(self):
        """Парсинг конфигурации"""
        # Основные параметры
        self.audio_folder           = self.config["OPTIONS"]["sources_dir"]
        self.engine_name            = self.config["OPTIONS"].get("transcribe_engine", "openai-whisper").strip()
        self.whisper_model          = self.config["OPTIONS"].get("whisper_model", "tiny").strip()
        self.use_cuda = Helper.parse_bool(self.config["OPTIONS"].get("use_cuda", "1"), default=True)
        self.text_language          = self.config["OPTIONS"].get("force_transcribe_language", "").strip()
        self.text_language          = self.text_language if self.text_language else None
        self.model_path             = self.config["OPTIONS"].get("model_path", "./models/").strip()
        self.skip_transcoded_files  = Helper.parse_bool(self.config["OPTIONS"].get("skip_transcoded_files"), default=False)

        self.export_srt_file        = Helper.parse_bool(self.config["OPTIONS"].get("export_srt_file"), default=False)
        self.export_raw_file        = Helper.parse_bool(self.config["OPTIONS"].get("export_raw_file"), default=False)

        self.enable_logging = Helper.parse_bool(self.config["OPTIONS"].get("logging", "0"), default=False)

#        self.decode_to_wav          = self.config["OPTIONS"].get("decode_to_wav", "0") == "1"
#        self.max_workers            = int(config["OPTIONS"].get("max_workers", "1"))

        if not self.engine_name or not self.whisper_model:
            print("❌ Required parameters missing in settings.ini:")
            print("   - transcribe_engine (faster-whisper or openai-whisper)")
            print("   - whisper_model (tiny, base, small, medium, large)")
            print("\nUsage: python3 {os.path.basename(sys.argv[0])}")
            sys.exit(1)
        
        # Параметры транскрипции
        self._parse_transcribe_params()

    def _parse_transcribe_params(self):
        """Парсинг параметров транскрипции"""
        transcribe_section = self.config["TRANSCRIBE"]
        
        self.beam_size = Helper.parse_int(transcribe_section.get("beam_size"), default=None)
        
        temperature_raw = transcribe_section.get("temperature", "").strip()
        if ',' in temperature_raw:
            self.temperature = Helper.parse_list_of_floats(temperature_raw, default=None)
        else:
            self.temperature = Helper.parse_float(temperature_raw, default=None)
            
        self.condition_on_prev_tokens = Helper.parse_bool(transcribe_section.get("condition_on_prev_tokens"), default=None)
        self.initial_prompt = transcribe_section.get("initial_prompt", "").strip() or None
        self.compression_ratio_threshold = Helper.parse_float(transcribe_section.get("compression_ratio_threshold"), default=None)
        self.logprob_threshold = Helper.parse_float(transcribe_section.get("logprob_threshold"), default=None)
        self.no_speech_threshold = Helper.parse_float(transcribe_section.get("no_speech_threshold"), default=None)
        self.patience = Helper.parse_float(transcribe_section.get("patience"), default=None)
        self.length_penalty = Helper.parse_float(transcribe_section.get("length_penalty"), default=None)
        self.suppress_blank = Helper.parse_bool(transcribe_section.get("suppress_blank"), default=None)
        
        suppress_tokens_raw = transcribe_section.get("suppress_tokens", "").strip()
        self.suppress_tokens = Helper.parse_list_of_ints(suppress_tokens_raw, default=None)
        
        self.without_timestamps = Helper.parse_bool(transcribe_section.get("without_timestamps"), default=None)
        self.max_initial_timestamp = Helper.parse_float(transcribe_section.get("max_initial_timestamp"), default=None)
        self.fp16 = Helper.parse_bool(transcribe_section.get("fp16"), default=None)
        self.temperature_increment_on_fallback = Helper.parse_float(transcribe_section.get("temperature_increment_on_fallback"), default=None)

# --------------------------------------------------------------------------------------------- #

class AudioProcessor:
    """Класс для обработки аудиофайлов"""
    
    def __init__(self, config):
        self.config = config
        self.audio_exts = ['mp3', 'aac', 'ogg', 'wav', 'opus', 'flac', 'm4a', 'wma', 'aiff', 'amr']
        self.model = None
        self.start_time = None

        # Статистика
        self.total_duration = 0.0
        self.total_processing_time = 0.0
        self.processed_files_count = 0
        self.successful_files_count = 0
        self.failed_files_count = 0

        # Поля для хранения путей текущего файла
        self.current_audio_path = None
        self.current_dirname = None
        self.current_basename = None
        self.current_name_noext = None
        self.current_relative_path = None
        self.current_timecode_file = None
        self.current_rawtext_file = None
        self.current_error_file = None
        self.current_srt_file = None

        # Настройка логгирования
        if self.config.enable_logging:
            self._setup_logging()

    def _setup_logging(self):
        """Настройка логгирования"""
        logging.basicConfig(
            filename='transcription.log',
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            encoding='utf-8'
        )

    def _log_message(self, message):
        """Логгирование сообщения"""
        if self.config.enable_logging:
            logging.info(message)

    def _setup_file_paths(self, audio_path):
        """Настройка путей для текущего аудиофайла"""
        self.current_audio_path = audio_path
        self.current_dirname = os.path.dirname(audio_path)
        self.current_basename = os.path.basename(audio_path)
        self.current_name_noext = os.path.splitext(self.current_basename)[0]
        self.current_relative_path = Path(audio_path).relative_to(self.current_dirname)

        self.current_timecode_file = os.path.join(self.current_dirname, self.current_name_noext + '_timecodes.txt')
        self.current_rawtext_file = os.path.join(self.current_dirname, self.current_name_noext + '_raw.txt')
        self.current_srt_file = os.path.join(self.current_dirname, self.current_name_noext + '.srt')
        self.current_error_file = os.path.join(self.current_dirname, self.current_name_noext + '_ERROR.txt')

    def initialize_engine(self):
        """Инициализация движка транскрипции и загрузка модели"""
        engine_name = self.config.engine_name
        whisper_model = self.config.whisper_model
        model_path = os.path.join(self.config.model_path, "openai-whisper")

        # Определяем устройство для обработки
        if self.config.use_cuda and torch.cuda.is_available():
            device = "cuda"
            print("🚀 Using CUDA for processing")
        else:
            device = "cpu"
            self.config.fp16 = False
            if self.config.use_cuda:
                print("⚠️  CUDA enabled in config but not available, using CPU")
            else:
                print("💻 Using CPU for processing")

        print()

        if engine_name == "openai-whisper":
            import whisper
            print(f'Loading model: "{whisper_model}" using engine: {engine_name}...')
            self.model = whisper.load_model(
                whisper_model,
                device=device,
                download_root=model_path
            )
        else:
            raise ValueError(f"Unknown transcribe_engine: '{engine_name}'. Use 'openai-whisper'.")

        print('✅ Model loaded.\n')
        self._log_session_start(engine_name, whisper_model, device)

    def should_skip_file(self, audio_path):
        """Проверка, нужно ли пропускать файл"""
        if not self.config.skip_transcoded_files:
            return False

        # Проверяем существование выходных файлов
        if os.path.exists(self.current_timecode_file) and os.path.exists(self.current_rawtext_file):
            # Проверяем, что выходные файлы новее исходного
            try:
                audio_mtime = os.path.getmtime(audio_path)
                timecode_mtime = os.path.getmtime(self.current_timecode_file)
                rawtext_mtime = os.path.getmtime(self.current_rawtext_file)

                # Файл можно пропустить, если оба выходных файла новее входного
                if timecode_mtime > audio_mtime and rawtext_mtime > audio_mtime:
                    return True
            except OSError:
                # Если возникла ошибка при получении времени модификации, не пропускаем файл
                return False

        return False

    def find_audio_files(self):
        """Поиск аудиофайлов в указанной директории"""
        audio_folder = self.config.audio_folder
        print(f'Scanning "{audio_folder}" (including subdirectories)...')
        
        if not os.path.exists(audio_folder):
            print(f'❌ Directory not found: "{audio_folder}"')
            return []
        
        audio_files = []
        for root, dirs, files in os.walk(audio_folder):
            for file in files:
                if Helper.match_ext(file, self.audio_exts):
                    full_path = os.path.join(root, file)
                    audio_files.append(full_path)
        
        return audio_files

    def get_file_info(self, audio_path):
        """Получение информации о файле: размер и длительность"""
        file_size = os.path.getsize(audio_path)

        try:
            audio = AudioSegment.from_file(audio_path)
            duration = len(audio) / 1000.0
            print(f'    Duration: {Helper.format_time(duration)}')
        except Exception:
            print(f'    ⚠️ Could not read duration: {audio_path}')
            duration = 0

        return file_size, duration

    def process_all_files(self):
        """Обработка всех аудиофайлов"""
        self.start_time = datetime.now()
        audio_files = self.find_audio_files()
        total_files = len(audio_files)
        
        if total_files == 0:
            print('No audio files found to process.')
            return
        
        print(f'✅ Found {total_files} audio file(s) to process.\n')
        
        # Инициализируем движок и загружаем модель
        self.initialize_engine()
        
        # Обработка файлов
        for idx, audio_file in enumerate(audio_files, 1):
            self._process_audiofile_openai_whisper(audio_file, idx, total_files)
        
        print('✅ All files processed.')
        # print(f'✅ Total time: {Helper.format_elapsed_time(datetime.now() - self.start_time)}')
        Helper.print_total_stats(self.processed_files_count, self.total_duration, self.total_processing_time)
        self._log_session_summary()
        print()
   
    def _process_audiofile_openai_whisper(self, audio_path, file_index, total_files):
        """Обработка аудиофайла с использованием openai-whisper"""
        # Проверяем, существует ли файл
        if not os.path.exists(audio_path):
            print(f'[{file_index:3d}/{total_files}] ❌ File not found: {audio_path}')
            return

        file_start_time = datetime.now()

        # Пути к выходным файлам
        self._setup_file_paths(audio_path)

        print(f'[{file_index:3d}/{total_files}] Processing: {self.current_relative_path}')

        # Проверка на пропуск уже обработанных файлов
        if self.should_skip_file(audio_path):
            print(f'[{file_index:3d}/{total_files}] ⏭️ Skipping (already processed)')
            print()
            return

        # Определяем длительность и размер файла
        filesize, duration = self.get_file_info(audio_path)

        try:
            # Подготовка параметров для openai-whisper
            transcribe_kwargs = {
                "verbose": False,
                "language": self.config.text_language,
                "temperature": self.config.temperature,
                "condition_on_previous_text": self.config.condition_on_prev_tokens,
                "initial_prompt": self.config.initial_prompt,
                "compression_ratio_threshold": self.config.compression_ratio_threshold,
                "logprob_threshold": self.config.logprob_threshold,
                "no_speech_threshold": self.config.no_speech_threshold,
                "patience": self.config.patience,
                "length_penalty": self.config.length_penalty,
                "suppress_blank": self.config.suppress_blank,
                "suppress_tokens": self.config.suppress_tokens,
                "without_timestamps": self.config.without_timestamps,
                "max_initial_timestamp": self.config.max_initial_timestamp,
                "fp16": self.config.fp16,
            }
            
            # Если beam_size задано, передаем его
            if self.config.beam_size is not None:
                transcribe_kwargs["beam_size"] = self.config.beam_size
            
            # Удаляем ключи со значением None
            transcribe_kwargs = {k: v for k, v in transcribe_kwargs.items() if v is not None}
            
            print(f"    Starting transcription with openai-whisper (model: {self.config.whisper_model})...")
            result = self.model.transcribe(audio_path, **transcribe_kwargs)
            
            # Оценка длительности
            if duration == 0 and 'segments' in result and len(result['segments']) > 0:
                duration = result['segments'][-1]['end']
                print("    Transcribing... (duration estimated)")
            else:
                print(f"    Transcribing... (duration: {Helper.format_time(duration)})")

            full_text = Helper.save_timecode_file(result, self.current_timecode_file)

#           скрываем избыточный progress bar с прогрессом
#            if duration > 0:
#                Helper.print_progress_bar(duration, duration)
#            print()
            
            # Сохраняем сырой текст
            if self.config.export_raw_file:
                Helper.save_text_files(full_text, self.current_rawtext_file)

            # Экспортируем SRT субтитры
            if self.config.export_srt_file:
                Helper.save_srt_file(result, self.current_srt_file)

            # Обновляем статистику
            processing_time = (datetime.now() - file_start_time).total_seconds()
            self.total_duration += duration
            self.total_processing_time += processing_time
            self.processed_files_count += 1
            self.successful_files_count += 1

            # Логгируем успешную обработку
            speed_ratio = processing_time / duration if duration > 0 else 0
            self._log_message(f"SUCCESS: {audio_path} | Size: {filesize} bytes | Duration: {Helper.format_time(duration)} | Time: {Helper.format_time(processing_time)} | Speed: {speed_ratio:.2f}x")

            print(f'✅ Done in {Helper.format_elapsed_time(datetime.now() - file_start_time)}')
            print()
        
        except Exception as e:
            print(f'\n❌ Error processing {audio_path}: {e}')
            # Записываем ошибку в файл
            with open(self.current_error_file, 'w', encoding='UTF-8') as ef:
                ef.write(f"Error processing {audio_path}: {e}\n")

            # Обновляем статистику ошибок
            self.processed_files_count += 1
            self.failed_files_count += 1

            # Логгируем ошибку
            self._log_message(f"ERROR: {audio_path} | Size: {filesize} bytes | Duration: {Helper.format_time(duration)} | Error: {str(e)}")

    def _log_session_start(self, engine_name, model_name, device):
        """Логгирование информации о начале сессии"""
        if self.config.enable_logging:
            # Получаем информацию о GPU если используется CUDA
            if device == "cuda" and torch.cuda.is_available():
                gpu_info = f"{torch.cuda.get_device_name(0)} (CUDA {torch.version.cuda})"
            else:
                gpu_info = "None"

            session_info = f"""
        SESSION STARTED
          Engine: {engine_name}
          Model: {model_name}
          Device: {device}
          GPU: {gpu_info}
          CUDA available: {torch.cuda.is_available()}
        """
            self._log_message(session_info.strip())


    def _log_session_summary(self):
        """Логгирование итоговой статистики сессии"""
        if self.config.enable_logging and self.processed_files_count > 0:
            speed_ratio = self.total_duration / self.total_processing_time if self.total_processing_time > 0 else 0
            session_time = (datetime.now() - self.start_time).total_seconds()

            summary = f"""
    SESSION SUMMARY:
      Files processed: {self.processed_files_count} (Success: {self.successful_files_count}, Failed: {self.failed_files_count})
      Total audio duration: {Helper.format_time(self.total_duration)}
      Total processing time: {Helper.format_time(self.total_processing_time)}
      Session time: {Helper.format_time(session_time)}
      Overall speed ratio: {speed_ratio:.2f}x
    """
            self._log_message(summary.strip())


# --------------------------------------------------------------------------------------------- #

class AudioTranscriber:
    """Основной класс приложения для транскрипции аудио"""

    @staticmethod
    def _print_copyright():
        """Вывод информации о копирайте"""
        print("=" * 50)
        print("Audio Transcriber v1.0")
        print("Based on OpenAI Whisper")
        print("=" * 50)
        print()

    @staticmethod
    def _print_gpu_info():
        """Вывод информации о видеокарте и режиме CUDA"""
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_count = torch.cuda.device_count()
            print(f"🚀 CUDA enabled: {gpu_count} GPU(s) available")
            print(f"   GPU: {gpu_name}")
            print(f"   CUDA version: {torch.version.cuda}")
        else:
            print("💻 CUDA disabled: using CPU only")
        print()

    def __init__(self):
        self._print_copyright()
        self._print_gpu_info()
        self.config = ConfigParser()
        self.processor = AudioProcessor(self.config)

    def run(self):
        """Запуск приложения"""
        try:
            self.processor.process_all_files()
        except Exception as e:
            print(f"Ошибка при выполнении приложения: {e}")
            return False
        return True

# --------------------------------------------------------------------------------------------- #

# Глобальная переменная для отслеживания прерывания
shutdown_requested = False

def main():
    """Основная функция"""
    app = AudioTranscriber()
    app.run()


def signal_handler(sig, frame):
    global shutdown_requested
    print('\n\n⚠️ Shutdown requested. Finishing current tasks...')
    shutdown_requested = True

# Регистрация обработчика сигнала Ctrl+C
# signal.signal(signal.SIGINT, signal_handler)

# .entrypoint
if __name__ == '__main__':
    main()

# -eof- #