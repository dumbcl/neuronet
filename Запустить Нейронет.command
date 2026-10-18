#!/bin/zsh
cd -- "${0:A:h}" || exit 1
if [[ ! -x .venv/bin/python ]]; then
  echo "Сначала установите зависимости по инструкции в README.md"
  read '?Нажмите Enter для закрытия...'
  exit 1
fi
exec .venv/bin/python -m neuronet_lab.desktop
