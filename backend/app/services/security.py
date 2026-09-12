"""Security helpers: prompt-injection heuristics and output leak detection.

These are defense-in-depth heuristics, not a complete solution. Prompt
injection cannot be fully "solved" by pattern matching (see docs/security.md
for an explicit statement of this limitation). The primary defenses in this
project are architectural:

1. Retrieved documents are treated as DATA in the system prompt (see
   `app/services/rag.py::SYSTEM_PROMPT`) -- the model is instructed never to
   follow instructions embedded in knowledge-base content.
2. Tool authorization is enforced in application code
   (`app/tools/service.py`), using an `AuthContext` the LLM never controls --
   it cannot be bypassed by anything the LLM or a user says.
3. The LLM's output is constrained to a strict structured schema
   (`LLMGroundedAnswer`, `extra="forbid"`), which limits (but does not
   eliminate) what a compromised generation can express.

This module adds two narrow, best-effort checks on top of that:
"""

import re


_INJECTION_MARKERS: tuple[str, ...] = (
    "ignore previous instructions",
    "ignore all previous instructions",
    "ignore the above instructions",
    "disregard previous instructions",
    "disregard the system prompt",
    "reveal your system prompt",
    "show me your system prompt",
    "print your instructions",
    "you are now",
    "act as",
    "jailbreak",
    "developer mode",
    "new instructions:",
)

_WORD_PATTERN = re.compile(r"[a-z0-9']+")


def looks_like_prompt_injection(text: str) -> bool:
    """Best-effort heuristic marker scan.

    Intended for logging/observability (see docs/observability.md), not as a
    hard content filter -- it will both miss real attempts (false negatives)
    and flag innocuous text (false positives). Retrieved knowledge-base
    content containing these markers is still just DATA per the system
    prompt; this function does not change that. It exists so such attempts
    can be logged and monitored.
    """
    lowered = text.lower()
    return any(marker in lowered for marker in _INJECTION_MARKERS)


def contains_system_prompt_leak(
    candidate_text: str,
    system_prompt: str,
    *,
    min_shared_words: int = 8,
) -> bool:
    """Detects whether generated output reproduces a long run of the system prompt.

    A simple n-gram overlap check: if any run of `min_shared_words`
    consecutive words from `system_prompt` appears verbatim in
    `candidate_text`, treat it as a likely leak of internal instructions.
    This is a coarse defense-in-depth measure, not a guarantee -- a
    sufficiently paraphrased leak would not be caught.
    """
    candidate_words = _WORD_PATTERN.findall(candidate_text.lower())
    if len(candidate_words) < min_shared_words:
        candidate_ngrams: set[tuple[str, ...]] = set()
    else:
        candidate_ngrams = {
            tuple(candidate_words[i : i + min_shared_words])
            for i in range(len(candidate_words) - min_shared_words + 1)
        }

    prompt_words = _WORD_PATTERN.findall(system_prompt.lower())
    for i in range(len(prompt_words) - min_shared_words + 1):
        if tuple(prompt_words[i : i + min_shared_words]) in candidate_ngrams:
            return True
    return False
