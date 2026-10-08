import math
import re
from pathlib import Path

import yaml

SECURITY_SIGNAL_RE = re.compile(
    r"\b(?:security (?:incident|concerns?|breach|vulnerability)|data breach|"
    r"hacked|compromised|"
    r"(?:got into|accessed|broke into|took over|took control of|hijacked) "
    r"(?:(?:my|our|your|their|the|this|that|an?) )?"
    r"(?:(?:work|business|company|personal|merchant|payout|payroll) )?accounts?|"
    r"(?:gained|got) access to (?:(?:my|our|your|their|the|this|that|an?) )?"
    r"(?:(?:work|business|company|personal|merchant|payout|payroll) )?accounts?|"
    r"phishing|malware|ransomware|"
    r"unauthori[sz]ed access|"
    r"unauthori[sz]ed login|account takeover|credential theft|suspicious login|"
    r"(?:someone|somebody) (?:logged|signed) in(?:to| to) "
    r"(?:(?:my|our|your|their|the|this|that|an?) )?"
    r"(?:(?:work|business|company|personal|merchant|payout|payroll) )?accounts?|"
    r"accounts? (?:was|were|got|has been|have been|had been) "
    r"(?:taken over|hijacked)|"
    r"unrecognized (?:device|login|location)|unrecognised (?:device|login|location)|"
    r"unfamiliar (?:device|login|location)|unknown (?:device|login|location)|"
    r"(?:device|login|location).{0,40}\b(?:i|we)\s+(?:do not|don['’]t)\s+recognize|"
    r"changed (?:my |the )?recovery (?:email|phone))\b",
    re.IGNORECASE | re.DOTALL,
)
# Fail-safe detector: a false positive costs a human review, a false negative
# lets an instruction reach the model, so the original broad shape is kept and
# only two benign constructions are carved out with narrow lookaheads.
#
# Carve-out 1: "review" as a modifier of a customer-facing thing ("review
# request", "review link", ...).
# Carve-out 2: "category of/for my|our|your|the <product-ish noun>" when it is
# not followed by a ticket-ish noun and no "to"/"as" re-assignment follows.
# Anything aimed at the ticket ("category of my ticket", "of this ticket")
# still matches.
# All gaps are bounded ({0,N}); there are no nested unbounded quantifiers.
_REVIEW = (
    r"review(?!\s+(?:requests?|links?|pages?|emails?|forms?|invitations?|"
    r"invites?|widgets?|stars?|scores?|ratings?|sites?|platforms?)\b)"
)
_CATEGORY = (
    r"category(?!\s+(?:of|for)\s+(?:my|our|your|the)\s+"
    r"(?:(?:product|store|shop|online)\s+)?"
    r"(?:products?|listings?|items?|stores?|shops?|catalogs?|catalogues?|"
    r"pages?|inventory|menu)\b"
    r"(?!\s+(?:tickets?|cases?|requests?|issues?|messages?|inquir\w+)\b)"
    r"(?![^.\n]{0,40}\b(?:to|as|into)\b))"
)
_FIELD = (
    r"(?:confidence|certainty|human_review|" + _REVIEW + r"|queue|" + _CATEGORY + r")"
)
PROMPT_OVERRIDE_RE = re.compile(
    r"\b(?:ignore|disregard|forget|override|bypass)\b.{0,80}"
    r"\b(?:instructions?|rules?|prompt|directive|policy)\b|"
    r"\b" + _FIELD + r"\b.{0,40}"
    r"\b(?:should|must|needs? to|has to)\b|"
    r"\b(?:set|mark|change|route|classify|place|send|assign|select)\b.{0,60}"
    r"\b" + _FIELD + r"\b|"
    r"\bdo not (?:flag|review|escalate)\b|"
    # Additive coverage: role/system markers and chat-template tokens.
    r"```\s*(?:system|assistant|developer)\b|"
    r"###\s*(?:instructions?|system)\b|"
    r"\[/?(?:system|inst)\]|<\|[a-z_]{1,30}\|>|"
    r"(?:^|\n)[ \t]*(?:system|assistant|developer)[ \t]*:[ \t]*"
    r"(?:you|ignore|disregard|forget|new|from now|always|never|set|classify|"
    r"output|respond|return)\b|"
    r"\byou are now\b|\bact as (?:a|an|if|the)\b|\bpretend (?:you are|you['’]re|to be)\b|"
    r"\bnew (?:rules?|instructions?)[ \t]*:|\bsystem (?:prompt|override)\b|"
    r"\b(?:i )?instruct you\b|\byou must (?:output|respond|return|answer)\b|"
    r"\brespond only with\b|\bdo not second-guess\b|\boverride[ \t]*:|"
    r"\b(?:classify|label|categori[sz]e|output)\s+(?:this|it|the ticket)\s+as\b|"
    r"\bno human review (?:required|needed|necessary)\b|"
    r"\b(?:all|any)\s+(?:previous|prior|above|earlier)\s+(?:instructions?|rules?|prompts?)\b|"
    r"\b(?:ignore|disregard|forget)\W+(?:what|everything|anything)\W+you\b",
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
        # customer_type is caller-supplied and unauthenticated, so it must never
        # raise confidence or suppress review.
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
