from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalSetup:
    system_guidelines: str


def _none_setup() -> RetrievalSetup:
    return RetrievalSetup(system_guidelines="")


def _bm25_setup() -> RetrievalSetup:
    return RetrievalSetup(
        system_guidelines=(
            "Retrieval mode: BM25. Relevant files have been suggested based on the task description. "
            "Start by examining the suggested files to understand the codebase structure."
        )
    )


def _bm25_rule_filter_setup() -> RetrievalSetup:
    return RetrievalSetup(
        system_guidelines=(
            "Retrieval mode: BM25 with rule filtering. Relevant rule files have been suggested. "
            "Focus on the suggested rule files that match the task description."
        )
    )


def _bm25_lint_aware_setup() -> RetrievalSetup:
    return RetrievalSetup(
        system_guidelines=(
            "Retrieval mode: BM25 with lint-aware localization. Files have been suggested based on "
            "both the task description and lint output analysis. Focus on the suggested files."
        )
    )


def _bm25_two_stage_setup() -> RetrievalSetup:
    return RetrievalSetup(
        system_guidelines=(
            "Retrieval mode: BM25 two-stage. Initial candidate rules have been identified, then "
            "focused retrieval performed. Examine the suggested files carefully."
        )
    )


def get_retrieval_setup(strategy: str) -> RetrievalSetup:
    s = strategy.lower().strip()
    if s in ("none", "", "off"):
        return _none_setup()
    if s == "bm25":
        return _bm25_setup()
    if s in ("bm25_rule_filter", "bm25_rule", "bm25-filter"):
        return _bm25_rule_filter_setup()
    if s in ("bm25_lint_aware", "bm25_lint", "bm25-lint"):
        return _bm25_lint_aware_setup()
    if s in ("bm25_two_stage", "bm25_2stage", "bm25-2stage"):
        return _bm25_two_stage_setup()
    return _none_setup()


def apply_retrieval_to_config(config: dict, strategy: str) -> dict:
    setup = get_retrieval_setup(strategy)
    if setup.system_guidelines:
        agent_cfg = config.setdefault("agent", {})
        base = agent_cfg.get("system_template", "")
        if setup.system_guidelines not in base:
            agent_cfg["system_template"] = (base + "\n\n" + setup.system_guidelines).strip()
    config.setdefault("run", {})["retrieval_strategy"] = strategy
    return config

