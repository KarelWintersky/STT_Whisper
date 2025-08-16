# SST_Whisper

Python wrapper for OpenAI's Whisper for processing all audio files in a specified folder and 
creating raw text + transcript with time stamps. 

See details in Russian in [README.ru.md](README.ru.md)

## Зависимости:

See [Whisper](https://github.com/openai/whisper) dependencies and installation instructions.


```
pip install -U openai-whisper
``` 

### Arch Linux

```bash
sudo pacman -S python
sudo pacman -S python-pip  
sudo pacman -S python-pydub 
sudo pacman -S python-configparser
sudo pacman -S python-openai-whisper
sudo pacman -Syu nvidia nvidia-utils cuda nvidia-open-dkms
<reboot>
sudo pacman -S python-pytorch-cuda
sudo pacman -S 

```


# ToDo

- Загружать модель только тогда, когда найден и поставлен в очередь распознавания не декодированный файл. То есть
  если все файлы декодированы - загружать модель смысла нет. 

- Многопоточность (`max_workers = 2`) делается через:

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
```

- Расцветка сообщений делается через:

```python
from colorama import init, Fore, Style
print(f"{Fore.RED}❌ Required parameters missing in settings.ini:{Style.RESET_ALL}")
```
- 
- декодировать в WAV? (`decode_to_wav = 0`)

