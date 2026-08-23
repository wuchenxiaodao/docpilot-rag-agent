"""Rule-based content guards for roadmap #11.

Two defenses that work without any external moderation model (so they function
in the local Qwen-only deployment, where the Groq-backed input ``Safeguard`` is
a no-op):

* **Retrieval injection defense** — scans retrieved tool output for
  instruction-override / prompt-extraction patterns and alerts (the durable
  defense is the "retrieved context is untrusted data" clause in the system
  prompt; this node adds detection + a per-request alert on top).
* **Output review** — scans the model's final answer for system-prompt leakage
  and signs of complying with an injected override, and alerts.

These are intentionally high-precision regex checks (few patterns, anchored) to
avoid false positives in a document-QA domain where the corpus may legitimately
discuss prompt-engineering or security topics. Auto-redaction would need a
moderation model — deferred to respect the no-new-dependency constraint.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Each pattern is (compiled regex, category). Case-insensitive. The
# combinatorial optional groups (all)(the)(previous|prior) let us match
# phrases like "ignore all previous instructions" without a brittle alternation.
_INJECTION_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"ignore\s+(?:all\s+)?(?:the\s+)?(?:previous\s+|prior\s+)?instructions", re.I), "override_instructions"),
    (re.compile(r"disregard\s+(?:all\s+)?(?:the\s+)?(?:previous\s+|prior\s+|above\s+)?(?:instructions|rules|directives)", re.I), "disregard_instructions"),
    (re.compile(r"forget\s+(?:all\s+)?(?:your\s+|the\s+)?(?:previous\s+|prior\s+)?(?:instructions|rules|role)", re.I), "forget_instructions"),
    (re.compile(r"(?:reveal|show|print|display)\s+(?:me\s+)?your\s+(?:system\s+)?(?:instructions|prompt|rules)", re.I), "prompt_extraction"),
    (re.compile(r"you\s+are\s+now\s+(?:a|an|the)\b", re.I), "role_override"),
    (re.compile(r"override\s+(?:your|the|system|all)\s+(?:instructions|rules|prompt)", re.I), "override_system"),
    (re.compile(r"new\s+(?:instructions|rules|directive)\s*:", re.I), "new_instructions"),
    (re.compile(r"stop\s+(?:following|using)\s+(?:your|the)\s+(?:instructions|rules)", re.I), "stop_rules"),
    (re.compile(r"\b(?:developer\s+mode|jailbreak|DAN)\b", re.I), "jailbreak"),
]

# Output-side: signs the answer leaked the system prompt or complied with an override.
_OUTPUT_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Verbatim signature of the DocPilot system prompt.
    (re.compile(r"You are DocPilot, a grounded knowledge assistant", re.I), "system_prompt_leak"),
    # The numbered RULES block from the system prompt, quoted back.
    (re.compile(r"^\s*RULES:\s*\n\s*1\.", re.I | re.M), "system_prompt_leak"),
    # Explicit compliance with an override request.
    (re.compile(r"as\s+(?:instructed|requested|you\s+asked)[,.]?\s*I\s+(?:ignored|disregarded|will\s+ignore)", re.I), "override_compliance"),
    (re.compile(r"I\s+am\s+now\s+acting\s+as\b", re.I), "role_compliance"),
]


@dataclass(frozen=True)
class GuardFinding:
    """A content-guard hit: where it came from and what category."""

    category: str
    snippet: str


def _scan(text: str, patterns: list[tuple[re.Pattern[str], str]]) -> list[GuardFinding]:
    findings: list[GuardFinding] = []
    if not text:
        return findings
    seen: set[str] = set()
    for rx, category in patterns:
        m = rx.search(text)
        if m is not None and category not in seen:
            seen.add(category)
            start = max(0, m.start() - 20)
            end = min(len(text), m.end() + 20)
            findings.append(GuardFinding(category=category, snippet=text[start:end].strip()))
    return findings


def scan_retrieval(text: str) -> list[GuardFinding]:
    """Scan retrieved tool output for prompt-injection / override patterns."""
    return _scan(text, _INJECTION_PATTERNS)


def scan_output(text: str) -> list[GuardFinding]:
    """Scan the model's final answer for leakage / override-compliance signs."""
    return _scan(text, _OUTPUT_PATTERNS)
