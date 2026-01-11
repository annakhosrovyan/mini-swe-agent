import json
import re
from pathlib import Path
from typing import Any

MAX_FILE_SIZE = 500_000
DEFAULT_TOP_K = 10
MAX_HELPER_FILES = 5


def run_bm25_retrieval(
    task: str,
    repo_path: Path,
    top_k: int = DEFAULT_TOP_K,
    rule_filter: bool = False,
    lint_output: str | None = None,
    two_stage: bool = False,
    file_extensions: list[str] | None = None,
    filter_pattern: str | None = None,
    rule_id_pattern: str | None = None,
) -> list[str]:
    if file_extensions is None:
        file_extensions = [".py"]
    
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        return []
    
    if not repo_path.exists():
        return []
    
    code_files = []
    for ext in file_extensions:
        pattern = f"*{ext}" if ext.startswith(".") else f"*.{ext}"
        code_files.extend(repo_path.rglob(pattern))
    python_files = code_files
    if rule_filter and filter_pattern:
        python_files = [f for f in python_files if re.search(filter_pattern, str(f))]
    
    if not python_files:
        return []
    
    file_contents = []
    valid_files = []
    for file_path in python_files:
        content = file_path.read_text(errors="ignore")
        if len(content) > MAX_FILE_SIZE:
            continue
        file_contents.append(content)
        valid_files.append(str(file_path.relative_to(repo_path)))
    
    if not file_contents:
        return []
    
    tokenized_docs = [doc.lower().split() for doc in file_contents]
    bm25 = BM25Okapi(tokenized_docs)
    
    query = task.lower().split()
    scores = bm25.get_scores(query)
    
    if lint_output and rule_id_pattern:
        rule_ids = re.findall(rule_id_pattern, lint_output)
        for i, file_path in enumerate(valid_files):
            for rule_id in rule_ids:
                if rule_id in file_path:
                    scores[i] *= 2.0
    
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    retrieved_files = [valid_files[i] for i in top_indices if scores[i] > 0]
    
    if two_stage and retrieved_files:
        if filter_pattern:
            rule_files = [f for f in retrieved_files if re.search(filter_pattern, f)]
        else:
            rule_files = retrieved_files
        if rule_files and rule_id_pattern:
            top_rule = rule_files[0]
            rule_extract_pattern = rule_id_pattern.replace("\\d+", "(\\d+)")
            rule_id = re.search(rule_extract_pattern, top_rule)
            if rule_id:
                rule_num = rule_id.group(1)
                rule_files_set = set(rule_files)
                helper_files = [f for f in valid_files if rule_num in f and f not in rule_files_set]
                retrieved_files = [top_rule] + helper_files[:MAX_HELPER_FILES]
    
    return retrieved_files


def run_retrieval_in_container(
    env: Any, 
    task: str, 
    strategy: str, 
    top_k: int = DEFAULT_TOP_K, 
    lint_output: str | None = None,
    file_extensions: list[str] | None = None,
    filter_pattern: str | None = None,
    rule_id_pattern: str | None = None,
) -> list[str]:
    if file_extensions is None:
        file_extensions = [".py"]
    
    install_cmd = "apt-get update -y >/dev/null 2>&1 && apt-get install -y python3-pip >/dev/null 2>&1 && python3 -m pip install -U rank-bm25 >/dev/null 2>&1 || true"
    env.execute(install_cmd)
    
    task_escaped = task.replace('"', '\\"').replace("$", "\\$")
    lint_output_escaped = (lint_output.replace('"', '\\"').replace("$", "\\$") if lint_output else "")
    extensions_str = json.dumps(file_extensions)
    filter_pattern_escaped = (filter_pattern.replace("\\", "\\\\").replace('"', '\\"').replace("$", "\\$") if filter_pattern else "")
    rule_id_pattern_escaped = (rule_id_pattern.replace("\\", "\\\\").replace('"', '\\"').replace("$", "\\$") if rule_id_pattern else "")
    
    retrieval_script = f'''python3 <<'BM25_EOF'
import json
import re
from pathlib import Path

MAX_FILE_SIZE = {MAX_FILE_SIZE}
MAX_HELPER_FILES = {MAX_HELPER_FILES}

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    print(json.dumps([]))
    exit(0)

task = """{task_escaped}"""
repo_path = Path("/testbed")
strategy = "{strategy}"
top_k = {top_k}
lint_output = """{lint_output_escaped}"""
file_extensions = {extensions_str}
rule_id_pattern = """{rule_id_pattern_escaped}""" if """{rule_id_pattern_escaped}""" else None

code_files = []
for ext in file_extensions:
    pattern = "*" + ext if ext.startswith(".") else "*." + ext
    code_files.extend(list(repo_path.rglob(pattern)))
python_files = code_files
rule_filter = strategy in ("bm25_rule_filter", "bm25_rule", "bm25-filter")
filter_pattern = """{filter_pattern_escaped}"""
if rule_filter and filter_pattern:
    python_files = [f for f in python_files if re.search(filter_pattern, str(f))]

if not python_files:
    print(json.dumps([]))
    exit(0)

file_contents = []
valid_files = []
for file_path in python_files:
    content = file_path.read_text(errors="ignore")
    if len(content) > MAX_FILE_SIZE:
        continue
    file_contents.append(content)
    valid_files.append(str(file_path.relative_to(repo_path)))

if not file_contents:
    print(json.dumps([]))
    exit(0)

tokenized_docs = [doc.lower().split() for doc in file_contents]
bm25 = BM25Okapi(tokenized_docs)
query = task.lower().split()
scores = bm25.get_scores(query)

lint_aware = strategy in ("bm25_lint_aware", "bm25_lint", "bm25-lint")
if lint_aware and lint_output and rule_id_pattern:
    rule_ids = re.findall(rule_id_pattern, lint_output)
    for i, file_path in enumerate(valid_files):
        for rule_id in rule_ids:
            if rule_id in file_path:
                scores[i] *= 2.0

two_stage = strategy in ("bm25_two_stage", "bm25_2stage", "bm25-2stage")
if two_stage:
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    retrieved_files = [valid_files[i] for i in top_indices if scores[i] > 0]
    if retrieved_files:
        filter_pattern = """{filter_pattern_escaped}"""
        if filter_pattern:
            rule_files = [f for f in retrieved_files if re.search(filter_pattern, f)]
        else:
            rule_files = retrieved_files
        if rule_files and rule_id_pattern:
            top_rule = rule_files[0]
            rule_extract_pattern = rule_id_pattern.replace("\\\\d+", "(\\\\d+)")
            rule_id = re.search(rule_extract_pattern, top_rule)
            if rule_id:
                rule_num = rule_id.group(1)
                rule_files_set = set(rule_files)
                helper_files = [f for f in valid_files if rule_num in f and f not in rule_files_set]
                retrieved_files = [top_rule] + helper_files[:MAX_HELPER_FILES]
    print(json.dumps(retrieved_files))
else:
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    retrieved_files = [valid_files[i] for i in top_indices if scores[i] > 0]
    print(json.dumps(retrieved_files))
BM25_EOF'''
    
    result = env.execute(retrieval_script)
    try:
        files = json.loads(result["output"].strip())
        return files if isinstance(files, list) else []
    except (json.JSONDecodeError, ValueError):
        return []
