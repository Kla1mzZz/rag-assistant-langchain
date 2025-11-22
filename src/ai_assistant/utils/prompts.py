from src.ai_assistant.core.config import config
from src.ai_assistant.core.logger import logger


def load_prompt(prompt_name: str) -> str:
    prompt_path = config.llm.prompts_dir / prompt_name

    if not prompt_path.exists():
        logger.warning(f"[Missing Prompt] {prompt_path} not found.")
        return "You are a helpful AI assistant."

    try:
        prompt_text = prompt_path.read_text(encoding="utf-8").strip()
        logger.info(f"[Loaded Prompt] {prompt_name}")
        return prompt_text
    except Exception as e:
        logger.error(f"[Prompt Load Error] {type(e).__name__}: {e}")
        return "You are a helpful AI assistant."
