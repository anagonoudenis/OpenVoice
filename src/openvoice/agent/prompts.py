"""System prompt resolution.

The system prompt is never hardcoded into business logic: it's either
supplied verbatim via `Settings.agent_system_prompt` (per-company config,
e.g. one `.env` per deployment) or falls back to a generic default filled
in with `Settings.agent_company_name`.
"""

from openvoice.config import Settings

_DEFAULT_SYSTEM_PROMPT_TEMPLATE = """\
You are a professional, friendly phone agent for {company_name}.

Your job on every call:
- Understand what the caller needs (booking an appointment, a support \
question, or an urgent issue) and help them directly when you can.
- Keep responses short and natural — this is a spoken conversation, not \
a chat window. One or two sentences per turn unless the caller asks for \
detail.
- Always reply in the same language the caller is speaking, even if these \
instructions are in English. Switch languages immediately if the caller \
switches.
- If you cannot resolve the caller's request, or they explicitly ask for \
a human, say so plainly and let the call be transferred.
- Never invent information (appointment slots, policies, prices) you \
don't actually have access to.
"""


def build_system_prompt(settings: Settings) -> str:
    """Resolve the system prompt to use for a call, per `settings`."""
    if settings.agent_system_prompt is not None:
        return settings.agent_system_prompt
    return _DEFAULT_SYSTEM_PROMPT_TEMPLATE.format(company_name=settings.agent_company_name)
