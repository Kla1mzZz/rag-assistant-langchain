FROM python:3.12

RUN pip install uv

ENV UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu

WORKDIR /app

COPY pyproject.toml ./
RUN uv lock --index-strategy unsafe-best-match
RUN uv sync --no-dev


COPY src/ src/
COPY prompts/ prompts/

EXPOSE 8000
CMD ["uv", "run", "uvicorn", "src.ai_assistant.main:app", "--host", "0.0.0.0", "--port", "8000"]