# server.py (FastAPI or your agent service)
from fastapi import FastAPI, Request
import logging
import uuid

from logging_config import init_logging, cv_request_id, cv_user, cv_ip

init_logging()
log = logging.getLogger(__name__)
app = FastAPI()


@app.middleware("http")
async def add_log_context(request: Request, call_next):
    # set per-request context vars so ContextFilter can include them
    token_req = cv_request_id.set(str(uuid.uuid4()))
    token_user = cv_user.set(getattr(request.state, "user_id", None))
    token_ip = cv_ip.set(request.client.host if request.client else None)
    try:
        response = await call_next(request)
        return response
    finally:
        # reset to avoid leaking between requests
        cv_request_id.reset(token_req)
        cv_user.reset(token_user)
        cv_ip.reset(token_ip)


@app.get("/healthz")
def health():
    log.debug("health check")
    return {"ok": True}
