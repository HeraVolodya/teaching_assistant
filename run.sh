#!/usr/bin/env bash
# Запуск «Асістента» в режимі розробки: бекенд + фронтенд однією командою.
#
#   ./run.sh          — режим заглушки: працює без жодної завантаженої моделі
#   ./run.sh --real    — реальні моделі через LM Studio
#   ./run.sh --stop    — зупинити все
#
# Ctrl+C зупиняє обидва сервери.

set -uo pipefail
cd "$(dirname "$0")"

API_PORT=8765
UI_PORT=5173
# DATA_DIR резолвиться нижче — він залежить від режиму:
#   заглушка → окремий тимчасовий каталог, який не шкода стерти;
#   --real    → платформний каталог застосунку, де лежать моделі й справжня БД.

free_port() {
  local port=$1 pids
  pids=$(lsof -nP -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)
  if [ -n "$pids" ]; then
    echo "  порт $port зайнятий (pid $pids) — зупиняю"
    kill -9 $pids 2>/dev/null || true
    sleep 1
  fi
}

if [ "${1:-}" = "--stop" ]; then
  echo "Зупиняю Асістент…"
  free_port "$API_PORT"
  free_port "$UI_PORT"
  echo "Зупинено."
  exit 0
fi

STUB=1
MODE="заглушка (без моделей)"
if [ "${1:-}" = "--real" ]; then
  STUB=0
  MODE="реальні моделі через LM Studio"
  if ! curl -s --max-time 2 http://127.0.0.1:1234/api/v1/models >/dev/null 2>&1; then
    echo "УВАГА: LM Studio не відповідає на 127.0.0.1:1234."
    echo "       Відкрийте LM Studio → Developer → увімкніть сервер, або запустіть:"
    echo "         ~/.lmstudio/bin/lms server start"
    echo "       Без нього генерація відповідей не працюватиме."
    echo
  fi
fi

# --- перевірка середовища ---
if [ ! -x backend/.venv/bin/python ]; then
  echo "Немає backend/.venv. Створюю…"
  python3.12 -m venv backend/.venv || { echo "Потрібен Python 3.12"; exit 1; }
  backend/.venv/bin/pip install -q --upgrade pip
  backend/.venv/bin/pip install -q -e "backend[dev]" || {
    echo "Не вдалося встановити залежності бекенду"; exit 1; }
fi
if [ ! -d frontend/node_modules ]; then
  echo "Немає frontend/node_modules. Встановлюю…"
  (cd frontend && npm install) || { echo "Потрібен Node.js/npm"; exit 1; }
fi

# --- каталог даних ---
# У режимі заглушки — окремий `~/.asistent-dev`: там сміттєва БД і жодних
# моделей, його не шкода стерти. У режимі `--real` беремо ПЛАТФОРМНИЙ каталог
# з app.config, бо саме туди `fetch_models.py` кладе моделі й саме там лежить
# справжня база. Інакше `--real` піднімався без жодної моделі й падав на
# першому ж питанні — при тому, що моделі на диску були.
if [ -n "${ASISTENT_DATA_DIR:-}" ]; then
  DATA_DIR="$ASISTENT_DATA_DIR"
elif [ "$STUB" = "1" ]; then
  DATA_DIR="$HOME/.asistent-dev"
else
  # PYTHONPATH=backend — пакет `app` не встановлений у venv як залежність,
  # uvicorn нижче отримує його через --app-dir.
  DATA_DIR=$(PYTHONPATH=backend backend/.venv/bin/python -c \
    'from app.config import Paths; print(Paths.resolve().data_dir)' 2>/dev/null) \
    || DATA_DIR=""
  if [ -z "$DATA_DIR" ]; then
    echo "Не вдалося визначити платформний каталог даних через app.config."
    exit 1
  fi
fi

free_port "$API_PORT"
free_port "$UI_PORT"

mkdir -p "$DATA_DIR"
echo
echo "  Режим:  $MODE"
echo "  Дані:   $DATA_DIR"
echo

cleanup() {
  echo
  echo "Зупиняю…"
  [ -n "${API_PID:-}" ] && kill "$API_PID" 2>/dev/null
  [ -n "${UI_PID:-}" ]  && kill "$UI_PID"  2>/dev/null
  wait 2>/dev/null
  exit 0
}
trap cleanup INT TERM

ASISTENT_STUB=$STUB ASISTENT_DATA_DIR="$DATA_DIR" \
  backend/.venv/bin/python -m uvicorn app.main:app \
  --app-dir backend --host 127.0.0.1 --port "$API_PORT" &
API_PID=$!

for i in $(seq 1 40); do
  curl -s --max-time 2 "http://127.0.0.1:$API_PORT/api/health" >/dev/null 2>&1 && break
  sleep 1
done

(cd frontend && npx vite --port "$UI_PORT" --strictPort --host 127.0.0.1) &
UI_PID=$!

for i in $(seq 1 40); do
  curl -s --max-time 2 "http://127.0.0.1:$UI_PORT/" >/dev/null 2>&1 && break
  sleep 1
done

echo
echo "─────────────────────────────────────────────"
echo "  Асістент готовий:  http://127.0.0.1:$UI_PORT"
echo "  API:               http://127.0.0.1:$API_PORT/api/health"
echo "  Документація API:  http://127.0.0.1:$API_PORT/docs"
echo
echo "  Ctrl+C — зупинити"
echo "─────────────────────────────────────────────"

wait
