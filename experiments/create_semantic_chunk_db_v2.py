"""DocPilot — v2 semantic chunk DB builder (corpus expansion).

Single-variable experiment: corpus size is the ONLY change.
- AcmeTech PDF goes through the ORIGINAL code path
  (PyPDFLoader -> split_by_sections from experiments.semantic_chunker),
  byte-identical logic, same source-path string as the v1 build.
- 9 markdown docs under docs/ go through the new explicit-config chunker
  (experiments.markdown_section_chunker). Heading levels and section title
  lists are hand-copied from the files, no heuristic detection.
- Same embedding model, same device, same normalization as v1.
- Writes to ./chroma_db_qwen3_semantic_chunks_v2. The v1 DB is never opened
  for writing; it is only read afterwards for COUNT + content-SHA verification.

Post-build checks (all printed, hard fail on any mismatch):
  1. per-document chunk counts and total
  2. chunk text-length distribution (min / median / max)
  3. 3 sampled chunks (chunk_id + first 80 chars)
  4. v1 DB COUNT still 9 (not polluted)
  5. every expected_chunk_ids entry from evals/eval_set.jsonl exists in v2
  6. content-level identity: for each of the 9 AcmeTech chunk_ids, the
     SHA-256 of page_content in v2 must equal that in v1
"""

import argparse
import hashlib
import json
import os
import random
import statistics
import sys

import chromadb
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.markdown_section_chunker import (
    MARKDOWN_SECTION_CONFIG,
    split_markdown_by_headings,
)
from experiments.semantic_chunker import split_by_sections

load_dotenv()

DB_V1 = "./chroma_db_qwen3_semantic_chunks"
DB_V2 = "./chroma_db_qwen3_semantic_chunks_v2"
DATA_DIR = "./data"
EVAL_SET = "./evals/eval_set.jsonl"
COLLECTION = "langchain"


def create_v2_db(rebuild: bool = False) -> None:
    model_path = os.path.join(
        os.path.expanduser("~"),
        "Models",
        "Qwen3-Embedding-0.6B",
    )
    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"Embedding model not found: {model_path}")

    if os.path.exists(DB_V2):
        if rebuild:
            import shutil

            shutil.rmtree(DB_V2)
            print(f"Rebuilding: deleted existing database at {DB_V2}")
        else:
            print(f"Database already exists at {DB_V2}. Use --rebuild to delete and recreate.")
            sys.exit(0)

    embeddings = HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},
    )

    chroma = Chroma(
        embedding_function=embeddings,
        persist_directory=DB_V2,
    )

    per_doc_counts: dict[str, int] = {}

    # ---- AcmeTech PDF: ORIGINAL code path, same join() source string as v1 ----
    for filename in sorted(os.listdir(DATA_DIR)):
        file_path = os.path.join(DATA_DIR, filename)
        if not filename.endswith(".pdf"):
            continue

        loader = PyPDFLoader(file_path)
        documents = loader.load()

        pages = [
            (doc.page_content, doc.metadata.get("page", 0) + 1, doc.metadata.get("source", ""))
            for doc in documents
        ]

        chunks = split_by_sections(pages)

        for chunk in chunks:
            chroma.add_documents(
                [Document(page_content=chunk["page_content"], metadata=chunk["metadata"])]
            )
        per_doc_counts[filename] = len(chunks)
        print(f"[acme-original-path] {filename}: {len(chunks)} chunks")

    # ---- Markdown docs: new explicit-config chunker ----
    dropped_by_doc: dict[str, list] = {}
    for md_path, cfg in MARKDOWN_SECTION_CONFIG.items():
        if not os.path.isfile(md_path):
            print(f"FATAL: configured markdown source missing: {md_path}")
            sys.exit(1)
        with open(md_path, encoding="utf-8-sig") as f:
            text = f.read()
        dropped: list = []
        try:
            chunks = split_markdown_by_headings(text, md_path, cfg, dropped_out=dropped)
        except ValueError as e:
            print(f"FATAL: markdown split failed: {e}")
            sys.exit(1)
        for chunk in chunks:
            chroma.add_documents(
                [Document(page_content=chunk["page_content"], metadata=chunk["metadata"])]
            )
        per_doc_counts[md_path] = len(chunks)
        dropped_by_doc[md_path] = [ln for ln in dropped if ln.strip()]
        print(
            f"[markdown-explicit]    {md_path}: {len(chunks)} chunks "
            f"(level={cfg['level']}, {len(cfg['sections'])} configured titles)"
        )

    print(f"\nSemantic chunk database v2 created and saved in {DB_V2}.")
    return per_doc_counts, dropped_by_doc


def verify(per_doc_counts: dict[str, int], dropped_by_doc: dict[str, list]) -> None:
    """Read-only post-build verification. Prints everything; exits 1 on mismatch."""
    failures = []

    v2 = chromadb.PersistentClient(path=DB_V2).get_collection(COLLECTION)
    all_v2 = v2.get(include=["metadatas", "documents"])
    meta_by_id = {m["chunk_id"]: m for m in all_v2["metadatas"]}
    doc_by_id = {m["chunk_id"]: d for m, d in zip(all_v2["metadatas"], all_v2["documents"])}

    # 1. counts
    total = v2.count()
    print("\n== 1. chunk counts ==")
    for name, n in per_doc_counts.items():
        print(f"  {name}: {n}")
    print(f"  TOTAL (collection.count()): {total}")

    # 1b. global chunk_id uniqueness (blocking)
    print("\n== 1b. global chunk_id uniqueness ==")
    all_ids = [m["chunk_id"] for m in all_v2["metadatas"]]
    uniq = len(set(all_ids))
    print(f"  total ids={len(all_ids)}, unique={uniq}")
    if uniq != len(all_ids) or uniq != total:
        from collections import Counter

        dupes = [cid for cid, n in Counter(all_ids).items() if n > 1]
        print(f"  FATAL: duplicate chunk_ids detected: {dupes}")
        failures.append(f"chunk_id uniqueness violated: {uniq} unique of {len(all_ids)}")
    else:
        print("  OK: all chunk_ids globally unique")

    # 2. text-length distribution
    lengths = sorted(len(d) for d in all_v2["documents"])
    med = statistics.median(lengths)
    print("\n== 2. chunk text length (chars) ==")
    print(f"  min={lengths[0]}  median={med:.0f}  max={lengths[-1]}")

    # 2b. short chunks (<100 chars), non-blocking but must be listed
    print("\n== 2b. short chunks (<100 chars) ==")
    shorts = [(cid, doc_by_id[cid]) for cid in sorted(doc_by_id) if len(doc_by_id[cid]) < 100]
    if shorts:
        print(f"  {len(shorts)} chunk(s) under 100 chars:")
        for cid, body in shorts:
            print(f"  --- {cid} ({len(body)} chars) ---")
            print(f"  {body!r}")
    else:
        print("  none")

    # 3. sample 3 chunks
    print("\n== 3. sample chunks ==")
    rng = random.Random(42)
    for cid in rng.sample(sorted(doc_by_id), 3):
        head = doc_by_id[cid][:80].replace("\n", " ")
        print(f"  {cid}\n    {head}")

    # 3b. known unindexed content (dropped before first configured heading)
    print("\n== 3b. known unindexed content (NOT in v2, must avoid in eval authoring) ==")
    any_dropped = False
    for md_path, lines in dropped_by_doc.items():
        if lines:
            any_dropped = True
            print(f"  --- {md_path}: {len(lines)} non-empty line(s) dropped ---")
            for ln in lines:
                print(f"    {ln}")
    if not any_dropped:
        print("  none")

    # 4. v1 DB still intact
    print("\n== 4. v1 DB integrity ==")
    v1 = chromadb.PersistentClient(path=DB_V1).get_collection(COLLECTION)
    v1_count = v1.count()
    print(f"  v1 COUNT = {v1_count} (expected 9)")
    if v1_count != 9:
        failures.append(f"v1 COUNT is {v1_count}, expected 9")

    # 5. every expected_chunk_ids from the eval set exists in v2
    print("\n== 5. eval-set expected_chunk_ids resolvable in v2 ==")
    expected = set()
    with open(EVAL_SET, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                expected.update(json.loads(line)["expected_chunk_ids"])
    missing = sorted(cid for cid in expected if cid not in meta_by_id)
    print(f"  {len(expected)} unique expected ids, missing in v2: {len(missing)}")
    for cid in missing:
        print(f"    MISSING: {cid}")
    if missing:
        failures.append(f"{len(missing)} expected chunk_ids missing in v2: {missing}")

    # 6. content-level identity of the 9 AcmeTech chunks (SHA-256 of page_content)
    print("\n== 6. AcmeTech content identity (v1 vs v2, SHA-256 of body) ==")
    all_v1 = v1.get(include=["metadatas", "documents"])
    v1_sha = {
        m["chunk_id"]: hashlib.sha256(d.encode("utf-8")).hexdigest()
        for m, d in zip(all_v1["metadatas"], all_v1["documents"])
    }
    acme_prefix = "AcmeTech_Employee_Handbook-"
    v2_acme = {cid: d for cid, d in doc_by_id.items() if cid.startswith(acme_prefix)}
    print(f"  v1 acme chunks: {len(v1_sha)}, v2 acme chunks: {len(v2_acme)}")
    if len(v2_acme) != len(v1_sha):
        failures.append(f"acme chunk count differs: v1={len(v1_sha)} v2={len(v2_acme)}")
    for cid in sorted(v1_sha):
        body = v2_acme.get(cid)
        if body is None:
            print(f"    MISMATCH: {cid} present in v1 but absent in v2")
            failures.append(f"{cid} absent in v2")
            continue
        sha_v2 = hashlib.sha256(body.encode("utf-8")).hexdigest()
        ok = sha_v2 == v1_sha[cid]
        print(f"    {'OK      ' if ok else 'MISMATCH'}: {cid}  sha={sha_v2[:12]}…")
        if not ok:
            failures.append(f"{cid} content SHA differs v1 vs v2")

    print("\n== verification summary ==")
    if failures:
        for f_ in failures:
            print(f"  FAIL: {f_}")
        sys.exit(1)
    print("  all checks passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create v2 semantic-chunk Chroma DB (corpus expansion)."
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Delete existing v2 database before rebuilding",
    )
    args = parser.parse_args()

    counts, dropped = create_v2_db(rebuild=args.rebuild)
    if counts is not None:
        verify(counts, dropped)
