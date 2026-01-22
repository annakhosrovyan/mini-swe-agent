import json
import re
from pathlib import Path
from typing import Any


MAX_FILE_SIZE = 500_000
DEFAULT_TOP_K = 10
DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_EMBEDDING_WEIGHT = 0.7
DEFAULT_BM25_WEIGHT = 0.3


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[tuple[str, int, int]]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text_slice = text[start:end]
        chunks.append((chunk_text_slice, start, end))
        start += chunk_size - overlap
        if start >= len(text):
            break
    return chunks


def run_retrieval_in_container(
    env: Any,
    task: str,
    strategy: str,
    top_k: int = DEFAULT_TOP_K,
    file_extensions: list[str] | None = None,
    index_all_files: bool = False,
    filter_pattern: str | None = None,
    source_path_prefix: str | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    embedding_weight: float = DEFAULT_EMBEDDING_WEIGHT,
    bm25_weight: float = DEFAULT_BM25_WEIGHT,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> list[dict[str, Any]]:
    install_cmd = "apt-get update -y >/dev/null 2>&1 && apt-get install -y python3-pip >/dev/null 2>&1 && python3 -m pip install -U rank-bm25 sentence-transformers faiss-cpu >/dev/null 2>&1 || true"
    env.execute(install_cmd)
    
    task_escaped = task.replace('"', '\\"').replace("$", "\\$").replace("\n", "\\n")
    extensions_str = json.dumps(file_extensions if file_extensions is not None else [".py"])
    filter_pattern_escaped = (filter_pattern.replace("\\", "\\\\").replace('"', '\\"').replace("$", "\\$") if filter_pattern else "")
    source_path_prefix_escaped = (source_path_prefix.replace("\\", "\\\\").replace('"', '\\"').replace("$", "\\$") if source_path_prefix else "")
    embedding_model_escaped = embedding_model.replace('"', '\\"').replace("$", "\\$")
    
    retrieval_script = rf'''python3 <<'HYBRID_EOF'
import json
import re
from pathlib import Path

MAX_FILE_SIZE = {MAX_FILE_SIZE}
CHUNK_SIZE = {chunk_size}
CHUNK_OVERLAP = {chunk_overlap}
EMBEDDING_WEIGHT = {embedding_weight}
BM25_WEIGHT = {bm25_weight}
TOP_K = {top_k}

def chunk_text(text, chunk_size, overlap):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text_slice = text[start:end]
        chunks.append((chunk_text_slice, start, end))
        start += chunk_size - overlap
        if start >= len(text):
            break
    return chunks

try:
    from rank_bm25 import BM25Okapi
    from sentence_transformers import SentenceTransformer
    import faiss
    import numpy as np
except ImportError:
    print(json.dumps([]))
    exit(0)

task = """{task_escaped}"""
repo_path = Path("/testbed")
file_extensions = {extensions_str}
index_all_files = {str(index_all_files).lower()}
source_path_prefix = """{source_path_prefix_escaped}""" if """{source_path_prefix_escaped}""" else None
embedding_model_name = """{embedding_model_escaped}"""

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

all_chunks = []
chunk_metadata = []
for file_path in filtered_files:
    try:
        content = file_path.read_text(errors="ignore")
        if len(content) > MAX_FILE_SIZE:
            continue
        file_rel_path = str(file_path.relative_to(repo_path))
        chunks = chunk_text(content, CHUNK_SIZE, CHUNK_OVERLAP)
        for chunk_text_slice, start, end in chunks:
            all_chunks.append(chunk_text_slice)
            chunk_metadata.append(dict(path=file_rel_path, start=start, end=end))
    except Exception:
        continue

if not all_chunks:
    print(json.dumps([]))
    exit(0)

model = SentenceTransformer(embedding_model_name)
task_embedding = model.encode([task])[0]
chunk_embeddings = model.encode(all_chunks)

dimension = chunk_embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
chunk_embeddings_normalized = chunk_embeddings / np.linalg.norm(chunk_embeddings, axis=1, keepdims=True)
task_embedding_normalized = task_embedding / np.linalg.norm(task_embedding)
index.add(chunk_embeddings_normalized.astype('float32'))

scores_embedding, indices_embedding = index.search(np.expand_dims(task_embedding_normalized.astype('float32'), 0), min(TOP_K * 2, len(all_chunks)))

tokenized_chunks = [chunk.lower().split() for chunk in all_chunks]
bm25 = BM25Okapi(tokenized_chunks)
query_tokens = task.lower().split()
scores_bm25 = bm25.get_scores(query_tokens)

normalized_scores_bm25 = (scores_bm25 - scores_bm25.min()) / (scores_bm25.max() - scores_bm25.min() + 1e-10)

hybrid_scores = {{}}
for i, (score_emb, idx) in enumerate(zip(scores_embedding[0], indices_embedding[0])):
    score_bm25 = normalized_scores_bm25[idx]
    hybrid_score = EMBEDDING_WEIGHT * float(score_emb) + BM25_WEIGHT * float(score_bm25)
    hybrid_scores[idx] = hybrid_score

all_indices_with_scores = [(idx, score) for idx, score in hybrid_scores.items() if score > 0]
all_indices_with_scores.sort(key=lambda x: x[1], reverse=True)
top_indices = [idx for idx, _ in all_indices_with_scores[:TOP_K]]

retrieved_chunks = []
for idx in top_indices:
    metadata = chunk_metadata[idx].copy()
    metadata["content"] = all_chunks[idx]
    retrieved_chunks.append(metadata)

print(json.dumps(retrieved_chunks))
HYBRID_EOF'''
    
    result = env.execute(retrieval_script)
    try:
        chunks = json.loads(result["output"].strip())
        return chunks if isinstance(chunks, list) else []
    except (json.JSONDecodeError, ValueError):
        return []
