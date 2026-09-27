import math
import re
from pathlib import Path

import yaml

SECURITY_SIGNAL_RE = re.compile(
    r"\b(?:security (?:incident|concerns?|breach|vulnerability)|data breach|"
    r"hacked|compromised|got into (?:my )?account|phishing|malware|ransomware|"
    r"unauthori[sz]ed access|"
    r"unauthori[sz]ed login|account takeover|credential theft|suspicious login|"
    r"someone logged into my account|somebody logged into my account|"
    r"unrecognized (?:device|login|location)|unrecognised (?:device|login|location)|"
    r"unfamiliar (?:device|login|location)|unknown (?:device|login|location)|"
    r"(?:device|login|location).{0,40}\b(?:i|we)\s+(?:do not|don['’]t)\s+recognize|"
    r"changed (?:my |the )?recovery (?:email|phone))\b",
    re.IGNORECASE | re.DOTALL,
)
PROMPT_OVERRIDE_RE = re.compile(
    r"\b(?:ignore|disregard|forget|override|bypass)\b.{0,80}"
    r"\b(?:instructions?|rules?|prompt|directive|policy)\b|"
    r"\b(?:set|mark|change|route|classify|place|send|assign|select)\b.{0,60}"
    r"\b(?:confidence|certainty|human_review|review|queue|category)\b|"
    r"\bdo not (?:flag|review|escalate)\b",
    re.IGNORECASE | re.DOTALL,
)


class RulesEngine:
    def __init__(self, config_path: str):
        self.path = Path(config_path)
        self.reload()

    def reload(self):
        with open(self.path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
        if not isinstance(loaded, dict):
            raise TypeError("Rules configuration must be a YAML mapping")
        self.cfg = loaded

    def snapshot(self) -> dict:
        """Return the active config mapping for a single request generation."""
        return self.cfg

    def configured_categories(self, config: dict | None = None) -> list[str]:
        """Return the category allow-list from a config snapshot."""
        values = (self.cfg if config is None else config).get("categories")
        if not isinstance(values, list):
            return []
        return [item for item in values if isinstance(item, str)]

    def apply(self, ticket: dict, model_out: object, config: dict | None = None) -> dict:
        # One snapshot per call: a concurrent reload must not let a single
        # classification read fields from two different config generations.
        cfg = self.cfg if config is None else config
        rules = cfg.get("rules", {})
        if not isinstance(rules, dict):
            rules = {}
        raw_forced = rules.get("forced_human_review_categories", [])
        forced_values = raw_forced if isinstance(raw_forced, (list, tuple, set)) else []
        forced = {item for item in forced_values if isinstance(item, str)}

        try:
            threshold = float(rules.get("confidence_threshold", 0.65))
        except (OverflowError, TypeError, ValueError):
            threshold = 0.65
        if not math.isfinite(threshold):
            threshold = 0.65
        threshold = min(max(threshold, 0.0), 1.0)

        try:
            enterprise_boost = float(rules.get("enterprise_confidence_boost", 0.0))
        except (OverflowError, TypeError, ValueError):
            enterprise_boost = 0.0
        if not math.isfinite(enterprise_boost):
            enterprise_boost = 0.0

        queue_map = cfg.get("queue_map", {})
        if not isinstance(queue_map, dict):
            queue_map = {}
        queue_map = {
            key: value
            for key, value in queue_map.items()
            if isinstance(key, str) and isinstance(value, str) and value.strip()
        }
        raw_categories = cfg.get("categories")
        categories = (
            [item for item in raw_categories if isinstance(item, str)]
            if isinstance(raw_categories, list)
            else []
        )
        enforce_categories = bool(categories)
        fallback_category = "Other / Needs Review"

        output = model_out if isinstance(model_out, dict) else {}
        ticket_text = " ".join(
            value for value in ticket.values() if isinstance(value, str)
        )
        security_signal = bool(SECURITY_SIGNAL_RE.search(ticket_text))
        prompt_override_signal = bool(PROMPT_OVERRIDE_RE.search(ticket_text))
        raw_category = output.get("category")
        category = raw_category.strip() if isinstance(raw_category, str) else ""
        invalid_category = not category or (
            enforce_categories and category not in categories
        )
        if security_signal:
            security_category = next(
                (
                    value
                    for value in categories
                    if value.casefold() == "security concerns"
                ),
                None,
            )
            category = security_category or fallback_category
            invalid_category = security_category is None
        elif prompt_override_signal:
            category = fallback_category
            invalid_category = True
        elif invalid_category:
            category = fallback_category

        raw_confidence = output.get("confidence")
        invalid_confidence = isinstance(raw_confidence, bool) or not isinstance(
            raw_confidence, (int, float)
        )
        if invalid_confidence:
            confidence = 0.0
        else:
            try:
                confidence = float(raw_confidence)
            except (OverflowError, TypeError, ValueError):
                confidence = 0.0
                invalid_confidence = True
        if not math.isfinite(confidence):
            confidence = 0.0
            invalid_confidence = True
        if (
            isinstance(ticket.get("customer_type"), str)
            and ticket["customer_type"].lower() == "enterprise"
        ):
            confidence += enterprise_boost
        confidence = min(max(confidence, 0.0), 1.0)

        raw_human_review = output.get("human_review")
        human_review = raw_human_review if isinstance(raw_human_review, bool) else True
        if (
            invalid_category
            or category in forced
            or confidence < threshold
            or security_signal
            or prompt_override_signal
            or invalid_confidence
        ):
            human_review = True

        reason = output.get("reason", "")
        if not isinstance(reason, str):
            reason = ""
        if security_signal:
            reason = "Potential security issue detected; routed for human review."
        elif prompt_override_signal:
            reason = "Ticket included classification instructions; routed for human review."
        queue = queue_map.get(category, "triage")
        return {
            "category": category,
            "confidence": confidence,
            "queue": queue,
            "reason": reason[:2_000],
            "human_review": human_review,
        }
