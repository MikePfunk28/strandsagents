# logging_config.py
import logging
import logging.config
import os
from contextvars import ContextVar

# request-scoped context you can set in middleware or task wrappers
cv_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)
cv_user:       ContextVar[str | None] = ContextVar("user", default=None)
cv_ip:         ContextVar[str | None] = ContextVar("ip", default=None)


class ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = cv_request_id.get()
        record.user = cv_user.get()
        record.ip = cv_ip.get()
        return True


def dict_config(env: str = "dev") -> dict:
    # K8s best practice: log to stdout/stderr, not files.
    # Switch formatter/levels by env var
    common_format = (
        "%(asctime)s %(levelname)s %(name)s "
        "req=%(request_id)s user=%(user)s ip=%(ip)s - %(message)s"
    )
    json_format = (
        '{"ts":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s",'
        '"req":"%(request_id)s","user":"%(user)s","ip":"%(ip)s","msg":"%(message)s"}'
    )

    return {
        "version": 1,
        "disable_existing_loggers": False,
        "filters": {
            "context": {"()": ContextFilter},
        },
        "formatters": {
            "console": {"format": common_format},
            "json": {"format": json_format},
        },
        "handlers": {
            "stdout": {
                "class": "logging.StreamHandler",
                "level": "DEBUG" if env == "dev" else "INFO",
                "formatter": "console" if env == "dev" else "json",
                "filters": ["context"],
                "stream": "ext://sys.stdout",
            },
        },
        "root": {
            "level": "DEBUG" if env == "dev" else "INFO",
            "handlers": ["stdout"],
        },
        # quiet noisy libs if you like:
        "loggers": {
            "uvicorn": {"level": "INFO", "handlers": ["stdout"], "propagate": False},
            "uvicorn.access": {"level": "INFO", "handlers": ["stdout"], "propagate": False},
            # add 'httpx', 'botocore', etc. as needed
        },
    }


def init_logging() -> None:
    env = os.getenv("APP_ENV", "dev")
    logging.config.dictConfig(dict_config(env))
