from pathlib import Path
import sys
from loguru import logger as loguru_logger


BASE_DIR = Path(__file__).resolve().parents[3]
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / "app.log"

LOGURU_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> "
    "| <level>{level: <8}</level> "
    "| <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> "
    "- <level>{message}</level>"
)


def setup_logger(level: str = "DEBUG"):
    loguru_logger.remove()
    loguru_logger.add(
        sys.stdout,
        colorize=True,
        format=LOGURU_FORMAT,
        level=level,
        enqueue=True,
    )
    loguru_logger.add(
        LOG_FILE,
        rotation="10 MB",
        retention="10 days",
        compression="zip",
        encoding="utf-8",
        format=LOGURU_FORMAT,
        level="INFO",
        enqueue=True,
        backtrace=True,
        diagnose=True,
    )
    loguru_logger.info(f"🚀 Loguru initialized. Logs → {LOG_FILE}")
    return loguru_logger


logger = setup_logger()
