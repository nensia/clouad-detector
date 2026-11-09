#!/bin/sh
# Запуск интерфейса на Linux и macOS:  sh run_linux_mac.sh
cd "$(dirname "$0")" || exit 1
if [ ! -d .venv ]; then
  echo "Первый запуск: создаю окружение и ставлю библиотеки. Это займёт несколько минут..."
  python3 -m venv .venv && . .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt
else
  . .venv/bin/activate
fi
python app.py
