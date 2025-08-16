# SST_Whisper

Python wrapper for OpenAI's Whisper for processing all audio files in a specified folder and 
creating raw text + transcript with time stamps. 

See details in Russian in [README.ru.md](README.ru.md)

## Dependencies

See [Whisper](https://github.com/openai/whisper) dependencies and installation instructions.

```
pip install -U openai-whisper
``` 
# OR
```
pip install git+https://github.com/openai/whisper.git
```

# OR, UPDATE

```
pip install --upgrade --no-deps --force-reinstall git+https://github.com/openai/whisper.git
``` 


# ToDo

- Многопоточность делается через

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
```

- Расцветка сообщений делается через

```python
from colorama import init, Fore, Style
print(f"{Fore.RED}❌ Required parameters missing in settings.ini:{Style.RESET_ALL}")
```

- Реализовать логгирование (`logging = 1`)
- Многопоточность `max_workers = 1`
- Включить/выключить CUDA `use_cuda = 1`
- декодировать в WAV? (`decode_to_wav = 0`)
- создавать .srt (`export_srt = 1`)

