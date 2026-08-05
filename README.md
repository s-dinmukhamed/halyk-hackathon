# Halyk AI Challenge — Covenant Compliance Agent

Автономный агент, который по кредитным документам заёмщиков определяет для каждого
**ковенанта**: соблюдён он или нарушен, число-обоснование и транзакцию-улику.

## ⚠️ Правила хакатона (важно)

- Все ответы генерирует **этот агент** (`run.py`). Ручное получение ответов —
  в т.ч. через готовые агенты (Claude Code, Codex) — **запрещено** и ведёт к дисквалификации.
  Такие инструменты используются только для разработки кода.
- Дедлайн сдачи: **9 авг 2026, 14:00 (Астана)**. Поздние отправки не принимаются.

## Стек (бесплатный, $0)

- **Gemini 2.0 Flash** — извлечение + vision-OCR (free tier)
- **Groq llama-3.3-70b** — быстрый текстовый fallback (free tier)
- **fastembed** — локальные эмбеддинги для поиска
- **Python** — все вычисления (детерминированно)

## Установка

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # вписать GEMINI_API_KEY и GROQ_API_KEY
```

Бесплатные ключи:
- Gemini: https://aistudio.google.com/apikey
- Groq: https://console.groq.com

## Запуск

```bash
python run.py --data ./data/public --out submission.json
```

Один прогон, без ручных шагов = «собственный агент участника».

## Архитектура

```
run.py
 1. INGEST    парсинг PDF (текст + таблицы; vision-fallback на сканы)
 2. EXTRACT   LLM → структура ковенантов (метрика, оператор, порог, дата, источник)
 3. RESOLVE   допсоглашения перекрывают договор по дате → актуальный порог
 4. COMPUTE   Python считает фактический показатель; скан реестра → транзакция-улика
 5. VERIFY    self-check: сверка числа с процитированным evidence
 6. EMIT      submission.json по шаблону; всегда заполнены все 3 поля
```

Принцип: **LLM извлекает — Python считает.** Числа никогда не считает модель.

## Структура

```
agent/
  config.py     — настройки из .env
  schemas.py    — pydantic-модели (Covenant, CovenantAnswer, Transaction, Submission)
  llm_client.py — Gemini + Groq, JSON-mode, ретраи/backoff
  ingest.py     — PDF → текст/таблицы/страницы-картинки
  extract.py    — документы → структурированные ковенанты
  resolve.py    — разрешение допсоглашений по effective_date
  compute.py    — детерминированные финансовые метрики + поиск транзакции-улики
  verify.py     — self-check ответов
  emit.py       — сборка submission.json
run.py          — CLI-оркестратор
tests/          — юнит-тесты (в т.ч. вычисления)
```
