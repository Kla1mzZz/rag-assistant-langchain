FROM python:3.12-slim

RUN pip install poetry

ENV POETRY_VIRTUALENVS_CREATE=false
ENV POETRY_NO_INTERACTION=1

WORKDIR /app

COPY pyproject.toml poetry.lock* ./
RUN poetry install --no-interaction --no-ansi --no-root

COPY src/ src/
COPY prompts/ prompts/

EXPOSE 8000
CMD ["poetry", "run", "uvicorn", "src.ai_assistant.main:app", "--host", "0.0.0.0", "--port", "8000"]