import json
import re
from pathlib import Path
from typing import Any

MAX_FILE_SIZE = 500_000
DEFAULT_TOP_K = 10
MAX_HELPER_FILES = 5


def run_retrieval_in_container(
    env: Any, 
    task: str, 
    strategy: str, 
    top_k: int = DEFAULT_TOP_K, 
    file_extensions: list[str] | None = None,
    index_all_files: bool = False,
    filter_pattern: str | None = None,
    source_path_prefix: str | None = None,
    entity_name: str | None = None,
) -> list[str]:
    install_cmd = "apt-get update -y >/dev/null 2>&1 && apt-get install -y python3-pip >/dev/null 2>&1 && python3 -m pip install -U rank-bm25 >/dev/null 2>&1 || true"
    env.execute(install_cmd)
    
    task_escaped = task.replace('"', '\\"').replace("$", "\\$")
    extensions_str = json.dumps(file_extensions if file_extensions is not None else [".py"])
    filter_pattern_escaped = (filter_pattern.replace("\\", "\\\\").replace('"', '\\"').replace("$", "\\$") if filter_pattern else "")
    source_path_prefix_escaped = (source_path_prefix.replace("\\", "\\\\").replace('"', '\\"').replace("$", "\\$") if source_path_prefix else "")
    entity_name_escaped = (entity_name.replace("\\", "\\\\").replace('"', '\\"').replace("$", "\\$") if entity_name else "")
    
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
file_extensions = {extensions_str}
index_all_files = {str(index_all_files).lower()}
source_path_prefix = """{source_path_prefix_escaped}""" if """{source_path_prefix_escaped}""" else None
entity_name = """{entity_name_escaped}""" if """{entity_name_escaped}""" else None

if index_all_files:
    code_files = [f for f in repo_path.rglob("*") if f.is_file()]
else:
    code_files = []
    for ext in file_extensions:
        pattern = "*" + ext if ext.startswith(".") else "*." + ext
        code_files.extend(list(repo_path.rglob(pattern)))
filtered_files = code_files
if source_path_prefix:
    filtered_files = [f for f in filtered_files if str(f.relative_to(repo_path)).startswith(source_path_prefix + "/") or str(f.relative_to(repo_path)) == source_path_prefix]
filter_pattern = """{filter_pattern_escaped}"""
if filter_pattern:
    filtered_files = [f for f in filtered_files if re.search(filter_pattern, str(f))]

if not filtered_files:
    print(json.dumps([]))
    exit(0)

file_contents = []
valid_files = []
for file_path in filtered_files:
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

two_stage = strategy == "bm25_two_stage"
if two_stage:
    candidate_top_k = 20
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:candidate_top_k]
    candidate_files = [(valid_files[i], scores[i]) for i in top_indices if scores[i] > 0]
    if entity_name and candidate_files:
        reranked_files = []
        entity_pattern = re.escape(entity_name)
        for file_path, bm25_score in candidate_files:
            file_content = None
            for i, vf in enumerate(valid_files):
                if vf == file_path:
                    file_content = file_contents[i]
                    break
            if file_content:
                score_boost = 0
                if re.search(rf"^\s*class\s+{entity_pattern}\s*[:\(]", file_content, re.MULTILINE):
                    score_boost = 100
                elif re.search(rf"^\s*def\s+{entity_pattern}\s*\(", file_content, re.MULTILINE):
                    score_boost = 100
                reranked_files.append((file_path, bm25_score + score_boost))
            else:
                reranked_files.append((file_path, bm25_score))
        reranked_files.sort(key=lambda x: x[1], reverse=True)
        retrieved_files = [f for f, _ in reranked_files[:3]]
    else:
        retrieved_files = [f for f, _ in candidate_files[:3]]
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
