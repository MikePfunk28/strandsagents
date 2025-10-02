from __future__ import annotations

from app.prompts.prompts import MENTOR_SYSTEM

from .base import BaseAssistant


class Mentor(BaseAssistant):
    ROLE = "coordinator"
    NAME = "mentor"
    PURPOSE = "Teach and supervise via tiny steps"
    SYSTEM = MENTOR_SYSTEM
    CAPABILITIES = ["think"]

