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


def _bm25_py_setup() -> RetrievalSetup:
    return RetrievalSetup(
        system_guidelines=(
            "Retrieval mode: BM25 (Python files only). Relevant Python files have been suggested based on the task description. "
            "Start by examining the suggested files to understand the codebase structure."
        )
    )


def _bm25_source_setup() -> RetrievalSetup:
    return RetrievalSetup(
        system_guidelines=(
            "Retrieval mode: BM25 (source code only). Relevant source files have been suggested based on the task description. "
            "Files have been filtered to focus on the main source code directory. Start by examining the suggested files to understand the codebase structure."
        )
    )


def _bm25_two_stage_setup() -> RetrievalSetup:
    return RetrievalSetup(
        system_guidelines=(
            "Retrieval mode: BM25 two-stage. Files have been suggested using a two-stage retrieval process: "
            "initial candidate generation followed by semantic reranking. Focus on the suggested files to understand the codebase structure."
        )
    )


def _hybrid_setup() -> RetrievalSetup:
    return RetrievalSetup(
        system_guidelines=(
            "Retrieval mode: Hybrid (embedding + BM25). Code chunks have been retrieved using a hybrid approach "
            "combining semantic embeddings and BM25 scoring. Focus on the suggested code chunks to understand the codebase structure."
        )
    )


def get_retrieval_setup(strategy: str) -> RetrievalSetup:
    s = strategy.lower().strip()
    if s in ("none", "", "off"):
        return _none_setup()
    if s == "bm25":
        return _bm25_setup()
    if s == "bm25_py":
        return _bm25_py_setup()
    if s == "bm25_source":
        return _bm25_source_setup()
    if s == "bm25_two_stage":
        return _bm25_two_stage_setup()
    if s == "hybrid":
        return _hybrid_setup()
    return _none_setup()


def apply_retrieval_to_config(config: dict, strategy: str) -> dict:
    setup = get_retrieval_setup(strategy)
    if setup.system_guidelines:
        agent_cfg = config.setdefault("agent", {})
        base = agent_cfg.get("system_template", "")
        if setup.system_guidelines not in base:
            agent_cfg["system_template"] = (base + "\n\n" + setup.system_guidelines).strip()
    run_cfg = config.setdefault("run", {})
    run_cfg["retrieval_strategy"] = strategy
    s = strategy.lower().strip()
    if s == "bm25":
        run_cfg["retrieval_index_all_files"] = True
    elif s == "bm25_py":
        run_cfg["retrieval_index_all_files"] = False
        run_cfg["retrieval_file_extensions"] = [".py"]
    elif s == "bm25_source":
        run_cfg["retrieval_index_all_files"] = False
        run_cfg["retrieval_file_extensions"] = [".py"]
    elif s == "bm25_two_stage":
        run_cfg["retrieval_index_all_files"] = False
        run_cfg["retrieval_file_extensions"] = [".py"]
    elif s == "hybrid":
        run_cfg["retrieval_index_all_files"] = True
    return config

