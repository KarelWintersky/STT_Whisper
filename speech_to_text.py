#!/usr/bin/python3
# All-in-one version

import io
import os
import re
import sys
import pwd
import subprocess
import xml.etree.ElementTree as ET
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


# ============================================================================================= #
#
#  GpuProcessGuard — проверка: не занята ли GPU другим Python-процессом
#
#  Алгоритм:
#    1. Вызывает nvidia-smi -q -x и парсит XML.
#    2. Находит все Compute-процессы с "python" в имени, кроме текущего PID.
#    3. Для каждого читает информацию из /proc/<pid> и выводит на экран.
#    4. Предлагает завершить процесс(ы) через kill -9 (sudo если нужно).
#    5. Если пользователь отказался — завершает текущий скрипт.
#
# ============================================================================================= #

class GpuProcessGuard:
    """Проверяет занятость GPU другими Python-процессами и предлагает их завершить."""

    def __init__(self, script_name=None):
        self.script_name = script_name or os.path.basename(sys.argv[0])
        self.current_pid = os.getpid()

    # ------------------------------------------------------------------ #
    #  Публичный метод
    # ------------------------------------------------------------------ #

    def check_and_resolve(self):
        """
        Вызвать при старте приложения.
        Если конкурирующих процессов нет — возвращает сразу.
        Если есть — показывает информацию и предлагает завершить.
        """
        conflicting = self._find_gpu_python_processes()
        if not conflicting:
            return

        print("⚠️  Обнаружены Python-процессы, использующие GPU:\n")
        for proc in conflicting:
            self._print_process_info(proc)

        print()
        if len(conflicting) == 1:
            prompt = "   Завершить этот процесс и продолжить? [y/N]: "
        else:
            prompt = f"   Завершить все {len(conflicting)} процесса(ов) и продолжить? [y/N]: "

        try:
            answer = input(prompt).strip().lower()
        except (EOFError, KeyboardInterrupt):
            answer = "n"

        if answer in ("y", "yes", "д", "да"):
            for proc in conflicting:
                self._kill_process(proc["pid"])
            print()
        else:
            print("\n❌ Запуск отменён пользователем.")
            sys.exit(0)

    # ------------------------------------------------------------------ #
    #  Получение процессов через nvidia-smi
    # ------------------------------------------------------------------ #

    def _find_gpu_python_processes(self):
        """
        Возвращает список словарей с информацией о Python-процессах на GPU.
        Каждый словарь: {pid, used_memory, proc_info}
        """
        try:
            result = subprocess.run(
                ["nvidia-smi", "-q", "-x"],
                capture_output=True, text=True, timeout=10
            )
        except FileNotFoundError:
            return []   # nvidia-smi недоступен — пропускаем проверку
        except subprocess.TimeoutExpired:
            print("⚠️  nvidia-smi не ответил за 10 секунд, пропускаем проверку GPU-процессов.")
            return []

        if result.returncode != 0:
            return []

        try:
            root = ET.fromstring(result.stdout)
        except ET.ParseError:
            return []

        found = []
        for gpu in root.findall("gpu"):
            processes_node = gpu.find("processes")
            if processes_node is None:
                continue
            for proc_node in processes_node.findall("process_info"):
                pid_text  = proc_node.findtext("pid",          "").strip()
                ptype     = proc_node.findtext("type",         "").strip()
                proc_name = proc_node.findtext("process_name", "").strip()
                used_mem  = proc_node.findtext("used_memory",  "N/A").strip()

                if not pid_text.isdigit():
                    continue
                pid = int(pid_text)

                if pid == self.current_pid:
                    continue
                if ptype != "C":            # только Compute-процессы
                    continue
                if "python" not in proc_name.lower():
                    continue

                found.append({
                    "pid":         pid,
                    "used_memory": used_mem,
                    "proc_info":   self._get_process_info(pid),
                })

        return found

    # ------------------------------------------------------------------ #
    #  Чтение информации о процессе из /proc
    # ------------------------------------------------------------------ #

    @staticmethod
    def _get_process_info(pid):
        """
        Читает информацию о процессе из /proc/<pid>.
        Возвращает словарь {name, user, cwd, cmdline}.
        При ошибке доступа поля содержат 'N/A'.
        """
        info = {"name": "N/A", "user": "N/A", "cwd": "N/A", "cmdline": "N/A"}
        proc_dir = f"/proc/{pid}"

        if not os.path.isdir(proc_dir):
            return info

        # Имя процесса
        try:
            with open(f"{proc_dir}/comm") as f:
                info["name"] = f.read().strip()
        except OSError:
            pass

        # Командная строка
        try:
            with open(f"{proc_dir}/cmdline", "rb") as f:
                raw = f.read()
            info["cmdline"] = raw.replace(b"\x00", b" ").decode("utf-8", errors="replace").strip() or "N/A"
        except OSError:
            pass

        # Рабочая директория
        try:
            info["cwd"] = os.readlink(f"{proc_dir}/cwd")
        except OSError:
            pass

        # Пользователь
        try:
            with open(f"{proc_dir}/status") as f:
                for line in f:
                    if line.startswith("Uid:"):
                        uid = int(line.split()[1])
                        try:
                            info["user"] = pwd.getpwuid(uid).pw_name
                        except KeyError:
                            info["user"] = str(uid)
                        break
        except OSError:
            pass

        return info

    # ------------------------------------------------------------------ #
    #  Вывод и завершение
    # ------------------------------------------------------------------ #

    @staticmethod
    def _format_memory(mem_str):
        """'9784 MiB' → '9784 MiB (9.55 GiB)'"""
        try:
            mib = float(mem_str.split()[0])
            return f"{int(mib)} MiB ({mib / 1024:.2f} GiB)"
        except (ValueError, IndexError):
            return mem_str

    def _print_process_info(self, proc):
        info    = proc["proc_info"]
        mem     = self._format_memory(proc["used_memory"])
        cmdline = info["cmdline"]
        if len(cmdline) > 80:
            cmdline = cmdline[:77] + "..."

        print(f"📊 Процесс PID: {proc['pid']}")
        print(f"   Память GPU:      {mem}")
        print(f"   Имя:             {info['name']}")
        print(f"   Пользователь:    {info['user']}")
        print(f"   Рабочая дирек.:  {info['cwd']}")
        print(f"   Команда:         {cmdline}")

    @staticmethod
    def _kill_process(pid):
        """Завершает процесс через kill -9, при нехватке прав — через sudo."""
        print(f"   🔫 Завершаем процесс PID {pid}...", end=" ", flush=True)

        try:
            result = subprocess.run(["kill", "-9", str(pid)], capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ Завершён.")
                return
        except FileNotFoundError:
            pass

        try:
            result = subprocess.run(["sudo", "kill", "-9", str(pid)], capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ Завершён (sudo).")
            else:
                print(f"❌ Не удалось завершить: {result.stderr.strip()}")
        except FileNotFoundError:
            print("❌ Команда kill не найдена.")


# ============================================================================================= #
#
#  ModelLoadProgress — прогресс загрузки модели openai-whisper с диска
#
#  Механизм: monkey-patch builtins.open
#    Перехватывает открытие конкретного .pt-файла модели и оборачивает его
#    в _ProgressRawFile — RawIOBase-обёртку, которая считает прочитанные байты
#    при каждом readinto(). torch.load() ничего не замечает — он просто читает
#    file-like object.
#
#  Размер файла берётся из таблицы WHISPER_MODEL_SIZES (известен заранее)
#  или с диска, если модель уже скачана.
#
#  Если .pt-файл ещё не существует (первая загрузка по сети) —
#  патч не активируется, загрузка идёт штатно без прогресса.
#
# ============================================================================================= #

class _ProgressRawFile(io.RawIOBase):
    """
    RawIOBase-обёртка над файлом модели.
    Вызывает callback(read_bytes, total_bytes) при каждом readinto().
    Используется внутри ModelLoadProgress — не создавать напрямую.
    """

    __slots__ = ('_f', '_total', '_pos', '_cb')

    def __init__(self, raw_file, total_size, callback):
        super().__init__()
        self._f     = raw_file
        self._total = total_size
        self._pos   = 0
        self._cb    = callback

    def readinto(self, b):
        n = self._f.readinto(b)
        if n:
            self._pos += n
            self._cb(self._pos, self._total)
        return n

    def readable(self):  return True
    def seekable(self):  return True

    def seek(self, pos, whence=0):
        result = self._f.seek(pos, whence)
        self._pos = result
        return result

    def tell(self):
        return self._f.tell()

    def close(self):
        if not self.closed:
            self._f.close()
        super().close()


class ModelLoadProgress:
    """
    Отображает прогресс загрузки модели openai-whisper с диска.

    Использование:
        progress = ModelLoadProgress()
        progress.wrap_whisper_load(model_name, path_to_pt_file)
        model = whisper.load_model(...)
        progress.unwrap_whisper_load()

    Если файл не найден на диске (модель ещё не скачана) —
    wrap_whisper_load() ничего не делает и загрузка идёт штатно.
    """

    BAR_WIDTH = 40

    # Известные размеры .pt-файлов openai-whisper (байты)
    WHISPER_MODEL_SIZES = {
        "tiny":      37_763_840,
        "tiny.en":   37_763_840,
        "base":      74_456_064,
        "base.en":   74_456_064,
        "small":    241_080_832,
        "small.en": 241_080_832,
        "medium":   764_354_560,
        "medium.en":764_354_560,
        "large":   2_952_790_016,
        "large-v1":2_952_790_016,
        "large-v2":2_952_790_016,
        "large-v3":2_952_790_016,
        "turbo":    809_582_592,
    }

    def __init__(self):
        self._last_pct      = -1
        self._original_open = None
        self._patched       = False

    # ------------------------------------------------------------------ #
    #  Публичный API
    # ------------------------------------------------------------------ #

    def wrap_whisper_load(self, model_name: str, model_file_path: str):
        """
        Активировать перехват builtins.open для указанного .pt-файла.
        Вызвать ДО whisper.load_model().
        """
        if not os.path.isfile(model_file_path):
            return

        expected = (
            self.WHISPER_MODEL_SIZES.get(model_name)
            or os.path.getsize(model_file_path)
        )

        self._print_bar(0, expected)
        self._install_patch(model_file_path, expected)

    def unwrap_whisper_load(self):
        """Снять перехват builtins.open после загрузки модели."""
        self._remove_patch()
        print()  # перевод строки после \r

    # ------------------------------------------------------------------ #
    #  Прогресс-бар
    # ------------------------------------------------------------------ #

    def _print_bar(self, current: int, total: int):
        if total <= 0:
            return
        pct = min(current / total * 100, 100.0)
        pct_int = int(pct)
        if pct_int == self._last_pct:
            return
        self._last_pct = pct_int

        filled  = int(self.BAR_WIDTH * pct / 100)
        bar     = '█' * filled + '░' * (self.BAR_WIDTH - filled)
        mb_cur  = current / 1024 / 1024
        mb_tot  = total   / 1024 / 1024
        print(f"\r    Model loading progress: |{bar}| {pct:5.1f}%  "
              f"({mb_cur:.0f} / {mb_tot:.0f} MB)",
              end='', flush=True)

    # ------------------------------------------------------------------ #
    #  Monkey-patch builtins.open
    # ------------------------------------------------------------------ #

    def _install_patch(self, target_path: str, expected_size: int):
        import builtins
        self._original_open = builtins.open
        self._patched       = True

        original   = self._original_open
        print_bar  = self._print_bar   # замыкание на метод

        def _patched_open(file, mode='r', *args, **kwargs):
            if (isinstance(file, (str, os.PathLike))
                    and os.path.abspath(str(file)) == os.path.abspath(target_path)
                    and 'b' in str(mode)):
                raw = _ProgressRawFile(
                    original(file, 'rb', buffering=0),
                    expected_size,
                    print_bar,
                )
                return io.BufferedReader(raw, buffer_size=512 * 1024)
            return original(file, mode, *args, **kwargs)

        builtins.open = _patched_open

    def _remove_patch(self):
        if self._patched and self._original_open is not None:
            import builtins
            builtins.open       = self._original_open
            self._original_open = None
            self._patched       = False


# ============================================================================================= #
#
#  Helper — вспомогательные статические методы
#
# ============================================================================================= #

class Helper:
    """Вспомогательный класс с утилитарными методами."""

    # ------------------------------------------------------------------ #
    #  НОВЫЙ МЕТОД: разрешение путей (./relative или /absolute)
    #  Работает одинаково на Linux и Windows.
    #  Относительные пути вычисляются от директории скрипта, а не от cwd.
    # ------------------------------------------------------------------ #
    @staticmethod
    def resolve_path(path_str):
        """
        Разрешает путь к файлу или директории:
          - Абсолютный путь (/home/... или C:\\...) — возвращает как есть (нормализованный).
          - Относительный (./sources, ../data, просто "sources") — вычисляется
            относительно директории запущенного скрипта (sys.argv[0]).

        Примеры в settings.ini:
            sources_dir = ./sources
            sources_dir = ../audio
            sources_dir = /mnt/nas/audio
            sources_dir = C:\\Users\\user\\audio
        """
        if not path_str:
            return path_str
        script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
        if os.path.isabs(path_str):
            return os.path.normpath(path_str)
        return os.path.normpath(os.path.join(script_dir, path_str))

    # ------------------------------------------------------------------ #
    #  НОВЫЙ МЕТОД: загрузка паттернов плохих слов из файла
    # ------------------------------------------------------------------ #
    @staticmethod
    def load_badwords_patterns(badwords_file_path):
        """
        Загружает список паттернов (plain text или regex) из файла.

        Формат файла:
          - Каждая непустая строка — отдельный паттерн.
          - Строки, начинающиеся с '#', считаются комментариями и пропускаются.
          - Поддерживаются полные регулярные выражения Python (re).

        Путь к файлу может быть:
          - Абсолютным: /home/user/project/settings_badwords.ini
          - Относительным от директории скрипта: ./settings_badwords.ini

        Возвращает список скомпилированных re.Pattern (с флагом IGNORECASE).
        При ошибке в regex — выводит предупреждение и пропускает паттерн.
        """
        patterns = []
        if not badwords_file_path:
            return patterns

        resolved = Helper.resolve_path(badwords_file_path)

        if not os.path.isfile(resolved):
            print(f"⚠️  badwords_file не найден: {resolved}")
            return patterns

        loaded_count = 0
        with open(resolved, encoding='utf-8') as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                # Пропускаем пустые строки и комментарии
                if not line or line.startswith('#'):
                    continue
                try:
                    patterns.append(re.compile(line, re.IGNORECASE))
                    loaded_count += 1
                except re.error as e:
                    print(f"⚠️  Неверный regex в badwords-файле (строка {lineno}): {line!r} → {e}")

        print(f"✅ Loaded {loaded_count} badword pattern(s) from: {os.path.basename(resolved)}")
        return patterns

    # ------------------------------------------------------------------ #
    #  ИЗМЕНЁННЫЙ МЕТОД: clean_text теперь принимает extra_patterns
    # ------------------------------------------------------------------ #
    @staticmethod
    def clean_text(text, extra_patterns=None):
        """
        Очистка текста от нежелательных фраз.

        Параметры:
          text           — исходный текст.
          extra_patterns — список скомпилированных re.Pattern, загруженных
                           из settings_badwords.ini. Если None или пустой —
                           используются только встроенные паттерны.

        Порядок применения:
          1. Встроенные паттерны (legacy-защита, работают всегда).
          2. Паттерны из файла (extra_patterns).
          3. Нормализация пробелов и пустых строк.
        """
        if not text:
            return text

        # Встроенные паттерны — оставлены как fallback на случай,
        # если badwords-файл не задан совсем.
        builtin_patterns = [
            r"Субтитры сделал DimaTorzok",
            r"Субтитры сделал\s+DimaTorzok",
            r"Субтитры создавал",
            r"Субтитры подогнал «Симон»",
            r"Субтитры добавил",
            r"DimaTorzok",
        ]

        cleaned = text
        for p in builtin_patterns:
            cleaned = re.sub(p, "", cleaned, flags=re.IGNORECASE)

        # Применяем паттерны из файла (уже скомпилированы)
        if extra_patterns:
            for pattern in extra_patterns:
                cleaned = re.sub(pattern, "", cleaned)

        cleaned = re.sub(r'\n\s*\n', '\n', cleaned)
        cleaned = re.sub(r' +', ' ', cleaned)
        return cleaned.strip()

    @staticmethod
    def format_time(seconds):
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    @staticmethod
    def format_elapsed_time(elapsed_time):
        return str(elapsed_time).split('.')[0]

    @staticmethod
    def format_srt_timestamp(seconds):
        hours        = int(seconds // 3600)
        minutes      = int((seconds % 3600) // 60)
        secs         = int(seconds % 60)
        milliseconds = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"

    @staticmethod
    def match_ext(filename, extensions):
        return filename.lower().split('.')[-1] in extensions

    @staticmethod
    def parse_float(value, default=None):
        if not value:
            return default
        try:
            return float(value)
        except ValueError:
            print(f"Предупреждение: Не удалось преобразовать '{value}' в float.")
            return default

    @staticmethod
    def parse_int(value, default=None):
        if not value:
            return default
        try:
            return int(value)
        except ValueError:
            print(f"Предупреждение: Не удалось преобразовать '{value}' в int.")
            return default

    @staticmethod
    def parse_bool(value, default=None):
        if not value:
            return default
        v = value.lower().strip()
        if v in ('true', '1', 'yes', 'on'):
            return True
        if v in ('false', '0', 'no', 'off'):
            return False
        print(f"Предупреждение: Не удалось преобразовать '{value}' в bool.")
        return default

    @staticmethod
    def parse_list_of_floats(value, default=None):
        if not value:
            return default
        try:
            return [float(x.strip()) for x in value.split(',') if x.strip()]
        except ValueError:
            print(f"Предупреждение: Не удалось преобразовать '{value}' в список float.")
            return default

    @staticmethod
    def parse_list_of_ints(value, default=None):
        if not value:
            return default
        try:
            return [int(x.strip()) for x in value.split(',') if x.strip()]
        except ValueError:
            print(f"Предупреждение: Не удалось преобразовать '{value}' в список int.")
            return default

    @staticmethod
    def print_total_stats(processed_files_count, total_duration, total_processing_time):
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
        if total <= 0:
            return
        fraction = current / total
        filled   = int(bar_length * fraction)
        bar      = '█' * filled + '░' * (bar_length - filled)
        print(f'\r    Progress: |{bar}| {fraction:.1%} ({Helper.format_time(current)}/{Helper.format_time(total)})',
              end='', flush=True)

    @staticmethod
    def save_timecode_file(result, timecode_file, extra_patterns=None):
        """extra_patterns пробрасывается в clean_text."""
        full_text = []
        with open(timecode_file, 'w', encoding='UTF-8') as f:
            for segment in result['segments']:
                start   = segment['start']
                text    = segment['text'].strip()
                cleaned = Helper.clean_text(text, extra_patterns)
                if cleaned:
                    hh, mm, ss = int(start // 3600), int((start % 3600) // 60), int(start % 60)
                    f.write(f"[{hh:02d}:{mm:02d}:{ss:02d}] {cleaned}\n")
                    full_text.append(cleaned)
        return full_text

    @staticmethod
    def save_text_files(full_text, rawtext_file):
        if not full_text:
            return
        rawtext = Helper.clean_text(' '.join(full_text))
        if not rawtext:
            return
        alltext = re.sub(r"([.!?])\s+", r"\1\n", rawtext).strip()
        with open(rawtext_file, 'w', encoding='UTF-8') as f:
            f.write(alltext)

    @staticmethod
    def save_srt_file(result, srt_file_path, extra_patterns=None):
        """extra_patterns пробрасывается в clean_text."""
        try:
            with open(srt_file_path, 'w', encoding='UTF-8') as f:
                for i, segment in enumerate(result['segments'], 1):
                    text    = segment['text'].strip()
                    cleaned = Helper.clean_text(text, extra_patterns)
                    if not cleaned:
                        continue
                    f.write(f"{i}\n")
                    f.write(f"{Helper.format_srt_timestamp(segment['start'])} --> "
                            f"{Helper.format_srt_timestamp(segment['end'])}\n")
                    f.write(f"{cleaned}\n\n")
            print(f"    SRT subtitles saved to: {os.path.basename(srt_file_path)}")
        except Exception as e:
            print(f"    ⚠️ Error exporting SRT: {e}")


# ============================================================================================= #
#
#  SSTLogger — логгирование операций транскрипции
#
# ============================================================================================= #

class SSTLogger:
    """Логгирует операции транскрипции в файл transcription.log."""

    def __init__(self, enable_logging=False):
        self.enable_logging = enable_logging
        if self.enable_logging:
            logging.basicConfig(
                filename='transcription.log',
                level=logging.INFO,
                format='%(asctime)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S',
                encoding='utf-8',
            )

    def log_message(self, message):
        if self.enable_logging:
            logging.info(message)

    def log_session_start(self, engine_name, model_name, device):
        if self.enable_logging:
            gpu_info = (
                f"{torch.cuda.get_device_name(0)} (CUDA {torch.version.cuda})"
                if device == "cuda" and torch.cuda.is_available()
                else "None"
            )
            self.log_message(
                f"SESSION STARTED | Engine: {engine_name} | Model: {model_name} | "
                f"Device: {device} | GPU: {gpu_info} | "
                f"CUDA available: {torch.cuda.is_available()}"
            )

    def log_session_summary(self, processed, successful, failed,
                            total_duration, total_processing_time, session_time):
        if self.enable_logging and processed > 0:
            ratio = total_duration / total_processing_time if total_processing_time > 0 else 0
            self.log_message(
                f"SESSION SUMMARY | Files: {processed} (OK:{successful} FAIL:{failed}) | "
                f"Audio: {Helper.format_time(total_duration)} | "
                f"Processing: {Helper.format_time(total_processing_time)} | "
                f"Session: {Helper.format_time(session_time)} | "
                f"Speed: {ratio:.2f}x"
            )


# ============================================================================================= #
#
#  AppConfigParser — парсинг settings.ini
#
#  ИЗМЕНЕНИЯ:
#    1. sources_dir теперь прогоняется через Helper.resolve_path() —
#       поддерживаются ./relative и /absolute пути на Linux и Windows.
#    2. Новый параметр badwords_file: путь к файлу с паттернами плохих слов.
#       Загружается через Helper.load_badwords_patterns().
#
# ============================================================================================= #

class AppConfigParser:
    """Читает и парсит конфигурацию из settings.ini."""

    def __init__(self, config_path="settings.ini"):
        self.config = configparser.ConfigParser()
        self.config.read(config_path, encoding='utf-8')
        self._parse_config()

    def _parse_config(self):
        opts = self.config["OPTIONS"]

        # ИЗМЕНЕНО: sources_dir теперь разрешается через Helper.resolve_path,
        # что позволяет использовать ./sources, ../audio и абсолютные пути.
        self.audio_folder  = Helper.resolve_path(opts["sources_dir"])

        self.engine_name   = opts.get("transcribe_engine", "openai-whisper").strip()
        self.whisper_model = opts.get("whisper_model", "tiny").strip()
        self.text_language = opts.get("force_transcribe_language", "").strip() or None
        self.model_path    = Helper.resolve_path(opts.get("model_path", "./models/").strip())

        self.skip_transcoded_files = Helper.parse_bool(
            opts.get("skip_transcoded_files"), default=False)
        self.use_cuda = Helper.parse_bool(
            opts.get("use_cuda", "1"), default=True)
        self.export_srt_file = Helper.parse_bool(
            opts.get("export_srt_file"), default=False)
        self.export_raw_file = Helper.parse_bool(
            opts.get("export_raw_file"), default=False)
        self.enable_logging = Helper.parse_bool(
            opts.get("logging", "0"), default=False)

        # НОВОЕ: загрузка паттернов плохих слов из внешнего файла.
        # В settings.ini задаётся как полный путь или относительный от скрипта:
        #   badwords_file = ./settings_badwords.ini
        #   badwords_file = /etc/transcriber/badwords.ini
        # Если параметр не задан — используются только встроенные паттерны clean_text.
        badwords_raw = opts.get("badwords_file", "").strip()
        self.badwords_patterns = (
            Helper.load_badwords_patterns(badwords_raw)
            if badwords_raw
            else []
        )

        if not self.engine_name or not self.whisper_model:
            print("❌ Required parameters missing in settings.ini:")
            print("   - transcribe_engine (openai-whisper)")
            print("   - whisper_model (tiny, base, small, medium, large, large-v3, ...)")
            sys.exit(1)

        self._parse_transcribe_params()

    def _parse_transcribe_params(self):
        s = self.config["TRANSCRIBE"]

        self.beam_size = Helper.parse_int(s.get("beam_size"), default=None)

        temp_raw = s.get("temperature", "").strip()
        self.temperature = (
            Helper.parse_list_of_floats(temp_raw, default=None)
            if ',' in temp_raw
            else Helper.parse_float(temp_raw, default=None)
        )

        self.condition_on_prev_tokens         = Helper.parse_bool(s.get("condition_on_prev_tokens"), default=None)
        self.initial_prompt                   = s.get("initial_prompt", "").strip() or None
        self.compression_ratio_threshold      = Helper.parse_float(s.get("compression_ratio_threshold"), default=None)
        self.logprob_threshold                = Helper.parse_float(s.get("logprob_threshold"), default=None)
        self.no_speech_threshold              = Helper.parse_float(s.get("no_speech_threshold"), default=None)
        self.patience                         = Helper.parse_float(s.get("patience"), default=None)
        self.length_penalty                   = Helper.parse_float(s.get("length_penalty"), default=None)
        self.suppress_blank                   = Helper.parse_bool(s.get("suppress_blank"), default=None)
        self.suppress_tokens                  = Helper.parse_list_of_ints(s.get("suppress_tokens", "").strip(), default=None)
        self.without_timestamps               = Helper.parse_bool(s.get("without_timestamps"), default=None)
        self.max_initial_timestamp            = Helper.parse_float(s.get("max_initial_timestamp"), default=None)
        self.fp16                             = Helper.parse_bool(s.get("fp16"), default=None)
        self.temperature_increment_on_fallback = Helper.parse_float(s.get("temperature_increment_on_fallback"), default=None)


# ============================================================================================= #
#
#  AudioProcessor — обработка аудиофайлов
#
#  ИЗМЕНЕНИЕ: все вызовы Helper.clean_text() теперь передают
#  self.config.badwords_patterns как extra_patterns.
#
# ============================================================================================= #

class AudioProcessor:
    """Находит аудиофайлы, загружает модель и выполняет транскрипцию."""

    def __init__(self, config):
        self.config     = config
        self.audio_exts = ['mp3', 'aac', 'ogg', 'wav', 'opus', 'flac', 'm4a', 'wma', 'aiff', 'amr']
        self.model      = None
        self.start_time = None

        self.total_duration        = 0.0
        self.total_processing_time = 0.0
        self.processed_files_count = 0
        self.successful_files_count = 0
        self.failed_files_count    = 0

        self.shutdown_requested = False

        self.current_audio_path    = None
        self.current_dirname       = None
        self.current_basename      = None
        self.current_name_noext    = None
        self.current_relative_path = None
        self.current_timecode_file = None
        self.current_rawtext_file  = None
        self.current_error_file    = None
        self.current_srt_file      = None

        self.logger       = SSTLogger(self.config.enable_logging)
        self.model_loaded = False

    def _setup_file_paths(self, audio_path):
        self.current_audio_path    = audio_path
        self.current_dirname       = os.path.dirname(audio_path)
        self.current_basename      = os.path.basename(audio_path)
        self.current_name_noext    = os.path.splitext(self.current_basename)[0]
        self.current_relative_path = Path(audio_path).relative_to(self.config.audio_folder)

        self.current_timecode_file = os.path.join(self.current_dirname, self.current_name_noext + '_timecodes.txt')
        self.current_rawtext_file  = os.path.join(self.current_dirname, self.current_name_noext + '_raw.txt')
        self.current_srt_file      = os.path.join(self.current_dirname, self.current_name_noext + '.srt')
        self.current_error_file    = os.path.join(self.current_dirname, self.current_name_noext + '_ERROR.txt')

    def initialize_engine(self):
        """Инициализация движка транскрипции и загрузка модели."""
        if self.model_loaded:
            return

        engine_name   = self.config.engine_name
        whisper_model = self.config.whisper_model
        model_path    = os.path.join(self.config.model_path, "openai-whisper")

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

            # Определяем путь к .pt-файлу чтобы показать прогресс загрузки
            try:
                pt_filename = (
                    os.path.basename(whisper._MODELS[whisper_model])
                    if hasattr(whisper, '_MODELS') and whisper_model in whisper._MODELS
                    else f"{whisper_model}.pt"
                )
                expected_pt = os.path.join(model_path, pt_filename)
            except Exception:
                expected_pt = os.path.join(model_path, f"{whisper_model}.pt")

            # Очищаем кэш GPU перед загрузкой модели
            if device == "cuda":
                torch.cuda.empty_cache()

            progress = ModelLoadProgress()
            progress.wrap_whisper_load(whisper_model, expected_pt)

            try:
                self.model = whisper.load_model(
                    whisper_model,
                    device=device,
                    download_root=model_path,
                )
            except torch.cuda.OutOfMemoryError:
                print(f"\n❌ Недостаточно GPU памяти для модели '{whisper_model}'.")
                props = torch.cuda.get_device_properties(0)
                print(f"   Доступно: {props.total_memory / 1024**3:.1f} GB")
                print("   Используйте модель меньшего размера (medium, small, tiny)")
                sys.exit(1)

            progress.unwrap_whisper_load()
            print('✅ Model loaded.\n')
        else:
            raise ValueError(
                f"Unknown transcribe_engine: '{engine_name}'. Use 'openai-whisper'."
            )

        self.model_loaded = True
        self.logger.log_session_start(engine_name, whisper_model, device)

    def should_skip_file(self, audio_path):
        if not self.config.skip_transcoded_files:
            return False
        if os.path.exists(self.current_timecode_file) and os.path.exists(self.current_rawtext_file):
            try:
                audio_mtime    = os.path.getmtime(audio_path)
                timecode_mtime = os.path.getmtime(self.current_timecode_file)
                rawtext_mtime  = os.path.getmtime(self.current_rawtext_file)
                if timecode_mtime > audio_mtime and rawtext_mtime > audio_mtime:
                    return True
            except OSError:
                return False
        return False

    def find_audio_files(self):
        audio_folder = self.config.audio_folder
        print(f'Scanning "{audio_folder}" (including subdirectories)...')
        if not os.path.exists(audio_folder):
            print(f'❌ Directory not found: "{audio_folder}"')
            return []
        audio_files = []
        for root, dirs, files in os.walk(audio_folder):
            for file in files:
                if Helper.match_ext(file, self.audio_exts):
                    audio_files.append(os.path.join(root, file))
        return audio_files

    def get_file_info(self, audio_path):
        file_size = os.path.getsize(audio_path)
        try:
            audio    = AudioSegment.from_file(audio_path)
            duration = len(audio) / 1000.0
            print(f'    Duration: {Helper.format_time(duration)}')
        except Exception:
            print(f'    ⚠️ Could not read duration: {audio_path}')
            duration = 0
        return file_size, duration

    def process_all_files(self):
        self.start_time  = datetime.now()
        audio_files      = self.find_audio_files()
        total_files      = len(audio_files)

        if total_files == 0:
            print('No audio files found to process.')
            return

        print(f'✅ Found {total_files} audio file(s) to process.\n')

        files_to_process = []
        for af in audio_files:
            self._setup_file_paths(af)
            if not self.should_skip_file(af):
                files_to_process.append(af)

        if not files_to_process:
            print('⏭️ All files already processed. Skipping model loading.')
            return

        print(f'Need to process {len(files_to_process)} file(s).\n')
        self.initialize_engine()

        for idx, audio_file in enumerate(audio_files, 1):
            if self.shutdown_requested:
                print("\n⚠️  Shutdown requested. Stopping processing...")
                break
            self._process_file(audio_file, idx, total_files)

        print('✅ All files processed.')
        Helper.print_total_stats(self.processed_files_count, self.total_duration, self.total_processing_time)
        self._log_session_summary()
        print()

    def _process_file(self, audio_path, file_index, total_files):
        if not os.path.exists(audio_path):
            print(f'[{file_index:3d}/{total_files}] ❌ File not found: {audio_path}')
            return

        file_start_time = datetime.now()
        self._setup_file_paths(audio_path)
        print(f'[{file_index:3d}/{total_files}] Processing: {self.current_relative_path}')

        if self.should_skip_file(audio_path):
            print(" " * 9 + '⏭️ Skipping (already processed)\n')
            return

        filesize, duration = self.get_file_info(audio_path)

        # Сокращение для удобства — паттерны из конфига
        bw = self.config.badwords_patterns

        try:
            transcribe_kwargs = {
                "verbose":                     False,
                "language":                    self.config.text_language,
                "temperature":                 self.config.temperature,
                "condition_on_previous_text":  self.config.condition_on_prev_tokens,
                "initial_prompt":              self.config.initial_prompt,
                "compression_ratio_threshold": self.config.compression_ratio_threshold,
                "logprob_threshold":           self.config.logprob_threshold,
                "no_speech_threshold":         self.config.no_speech_threshold,
                "patience":                    self.config.patience,
                "length_penalty":              self.config.length_penalty,
                "suppress_blank":              self.config.suppress_blank,
                "suppress_tokens":             self.config.suppress_tokens,
                "without_timestamps":          self.config.without_timestamps,
                "max_initial_timestamp":       self.config.max_initial_timestamp,
                "fp16":                        self.config.fp16,
            }
            if self.config.beam_size is not None:
                transcribe_kwargs["beam_size"] = self.config.beam_size

            # Удаляем ключи со значением None
            transcribe_kwargs = {k: v for k, v in transcribe_kwargs.items() if v is not None}

            print(f"    Starting transcription (model: {self.config.whisper_model})...")

            if self.shutdown_requested:
                print("    ⏹️  Processing cancelled by user")
                return

            result = self.model.transcribe(audio_path, **transcribe_kwargs)

            if self.shutdown_requested:
                print("    ⏹️  Processing cancelled by user")
                return

            # ИЗМЕНЕНО: clean_text получает паттерны из badwords-файла
            if 'segments' in result:
                for segment in result['segments']:
                    if 'text' in segment:
                        segment['text'] = Helper.clean_text(segment['text'], bw)

            if duration == 0 and 'segments' in result and result['segments']:
                duration = result['segments'][-1]['end']
                print("    Transcribing... (duration estimated)")
            else:
                print(f"    Transcribing... (duration: {Helper.format_time(duration)})")

            # ИЗМЕНЕНО: передаём bw во все save-методы
            full_text = Helper.save_timecode_file(result, self.current_timecode_file, bw)

            if self.config.export_raw_file and full_text:
                Helper.save_text_files(full_text, self.current_rawtext_file)

            if self.config.export_srt_file:
                Helper.save_srt_file(result, self.current_srt_file, bw)

            processing_time = (datetime.now() - file_start_time).total_seconds()
            self.total_duration         += duration
            self.total_processing_time  += processing_time
            self.processed_files_count  += 1
            self.successful_files_count += 1

            speed_ratio = processing_time / duration if duration > 0 else 0
            self.logger.log_message(
                f"SUCCESS: {audio_path} | Size: {filesize} bytes | "
                f"Duration: {Helper.format_time(duration)} | "
                f"Time: {Helper.format_time(processing_time)} | "
                f"Speed: {speed_ratio:.2f}x"
            )
            print(f'✅ Done in {Helper.format_elapsed_time(datetime.now() - file_start_time)}\n')

            # Освобождаем кэш GPU после каждого файла
            if self.config.use_cuda and torch.cuda.is_available():
                torch.cuda.empty_cache()

        except KeyboardInterrupt:
            print(f'\n⏹️  Processing cancelled by user for: {audio_path}')
            self.shutdown_requested = True
        except Exception as e:
            print(f'\n❌ Error processing {audio_path}: {e}')
            with open(self.current_error_file, 'w', encoding='UTF-8') as ef:
                ef.write(f"Error processing {audio_path}: {e}\n")
            self.processed_files_count += 1
            self.failed_files_count    += 1
            self.logger.log_message(
                f"ERROR: {audio_path} | Size: {filesize} bytes | "
                f"Duration: {Helper.format_time(duration)} | Error: {e}"
            )

    def _log_session_summary(self):
        if self.config.enable_logging and self.processed_files_count > 0:
            session_time = (datetime.now() - self.start_time).total_seconds()
            self.logger.log_session_summary(
                self.processed_files_count,
                self.successful_files_count,
                self.failed_files_count,
                self.total_duration,
                self.total_processing_time,
                session_time,
            )


# ============================================================================================= #
#
#  AudioTranscriber — точка входа приложения
#
# ============================================================================================= #

class AudioTranscriber:
    """Главный класс приложения. Инициализирует компоненты и запускает обработку."""

    @staticmethod
    def _print_copyright():
        print("=" * 50)
        print("Audio Transcriber v1.0")
        print("Based on OpenAI Whisper")
        print("=" * 50)
        print()

    @staticmethod
    def _print_gpu_info():
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            mem_total = props.total_memory / 1024**3
            print(f"🚀 CUDA enabled: {torch.cuda.device_count()} GPU(s) available")
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {mem_total:.1f} GB")
            print(f"   CUDA version: {torch.version.cuda}")
        else:
            print("💻 CUDA disabled: using CPU only")
        print()

    def __init__(self):
        self._print_copyright()

        # Проверяем, не занята ли GPU другим Python-процессом
        GpuProcessGuard().check_and_resolve()

        self._print_gpu_info()
        # ИЗМЕНЕНО: используем переименованный класс AppConfigParser
        # (переименование нужно чтобы не конфликтовать с импортированным configparser)
        self.config    = AppConfigParser()
        self.processor = AudioProcessor(self.config)

    def run(self):
        try:
            original_handler = signal.signal(signal.SIGINT, self._signal_handler)
            self.processor.process_all_files()
            signal.signal(signal.SIGINT, original_handler)
        except Exception as e:
            print(f"Ошибка при выполнении приложения: {e}")
            return False
        return True

    def _signal_handler(self, sig, frame):
        print('\n\n⚠️  Shutdown requested. Finishing current task...')
        self.processor.shutdown_requested = True


# ============================================================================================= #

def main():
    app     = AudioTranscriber()
    success = app.run()
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())

# -eof- #
