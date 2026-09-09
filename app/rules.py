import math
from pathlib import Path

import yaml


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

    def apply(self, ticket: dict, model_out: object) -> dict:
        rules = self.cfg.get("rules", {})
        if not isinstance(rules, dict):
            rules = {}
        raw_forced = rules.get("forced_human_review_categories", [])
        forced_values = raw_forced if isinstance(raw_forced, (list, tuple, set)) else []
        forced = {item for item in forced_values if isinstance(item, str)}

        try:
            threshold = float(rules.get("confidence_threshold", 0.65))
        except (TypeError, ValueError):
            threshold = 0.65
        if not math.isfinite(threshold):
            threshold = 0.65
        threshold = min(max(threshold, 0.0), 1.0)

        try:
            enterprise_boost = float(rules.get("enterprise_confidence_boost", 0.0))
        except (TypeError, ValueError):
            enterprise_boost = 0.0
        if not math.isfinite(enterprise_boost):
            enterprise_boost = 0.0

        queue_map = self.cfg.get("queue_map", {})
        if not isinstance(queue_map, dict):
            queue_map = {}
        queue_map = {
            key: value
            for key, value in queue_map.items()
            if isinstance(key, str) and isinstance(value, str) and value.strip()
        }
        configured_categories = self.cfg.get("categories")
        category_values = (
            configured_categories if isinstance(configured_categories, list) else []
        )
        categories = [item for item in category_values if isinstance(item, str)]
        enforce_categories = bool(categories)
        fallback_category = "Other / Needs Review"

        output = model_out if isinstance(model_out, dict) else {}
        raw_category = output.get("category")
        category = raw_category.strip() if isinstance(raw_category, str) else ""
        invalid_category = not category or (
            enforce_categories and category not in categories
        )
        if invalid_category:
            category = fallback_category

        raw_confidence = output.get("confidence", 0.0)
        try:
            confidence = float(raw_confidence)
        except (TypeError, ValueError):
            confidence = 0.0
        if not math.isfinite(confidence):
            confidence = 0.0
        if (
            isinstance(ticket.get("customer_type"), str)
            and ticket["customer_type"].lower() == "enterprise"
        ):
            confidence += enterprise_boost
        confidence = min(max(confidence, 0.0), 1.0)

        raw_human_review = output.get("human_review")
        human_review = raw_human_review if isinstance(raw_human_review, bool) else True
        if invalid_category or category in forced or confidence < threshold:
            human_review = True

        reason = output.get("reason", "")
        if not isinstance(reason, str):
            reason = ""
        queue = queue_map.get(category, "triage")
        return {
            "category": category,
            "confidence": confidence,
            "queue": queue,
            "reason": reason[:2_000],
            "human_review": human_review,
        }
