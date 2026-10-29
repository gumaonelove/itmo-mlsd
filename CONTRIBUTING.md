# Как работать с репозиторием

## Окружение

Зависимости управляются через [uv](https://docs.astral.sh/uv/), версии зафиксированы в `uv.lock`.

```bash
uv sync                     # создать .venv и поставить зависимости (включая dev-группу)
uv run pre-commit install   # поставить git-хук, который запускается перед каждым коммитом
```

## Линтеры и форматтеры

| Инструмент | Что делает | Где настроен |
| --- | --- | --- |
| [ruff](https://docs.astral.sh/ruff/) `check` | линтер: pycodestyle, pyflakes, isort, bugbear, pyupgrade, naming и др. | `pyproject.toml` → `[tool.ruff.lint]` |
| [ruff](https://docs.astral.sh/ruff/) `format` | форматтер, совместимый с black | `pyproject.toml` → `[tool.ruff.format]` |
| [pre-commit](https://pre-commit.com/) | запускает ruff и базовые проверки перед коммитом | `.pre-commit-config.yaml` |
| GitHub Actions | тот же pre-commit в CI на каждый пуш в любую ветку | `.github/workflows/lint.yml` |

Ручной запуск:

```bash
uv run ruff check . --fix     # линтер с автоисправлением
uv run ruff format .          # форматирование
uv run pre-commit run --all-files   # все хуки по всем файлам
```

Хуки pre-commit:

- `trailing-whitespace`, `end-of-file-fixer`, `mixed-line-ending`: пробелы и переводы строк;
- `check-yaml`, `check-toml`, `check-json`, `check-merge-conflict`, `check-case-conflict`: валидность конфигов и следы конфликтов;
- `check-added-large-files` (лимит 1 МБ): данные и веса моделей в git не храним;
- `detect-private-key`: защита от случайного коммита ключей;
- `ruff-check`, `ruff-format`: линтер и форматтер;
- `uv-lock`: `uv.lock` синхронизирован с `pyproject.toml`;
- `nbstripout`: из ноутбуков удаляются выводы ячеек.

Обновить версии хуков: `uv run pre-commit autoupdate`.

## Git-процесс

- `main` — только проверенное состояние, прямые пуши не делаем.
- Каждое домашнее задание делается в ветке `hw_<n>` и сдаётся через Pull Request в `main`.
- В PR добавляются ментор и два кросс-ревьюера; мерж после аппрува и зелёного CI.
- Коммиты — короткие, в повелительном наклонении: `Add design doc draft`, `Configure ruff`.
