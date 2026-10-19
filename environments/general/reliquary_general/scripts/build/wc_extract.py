import os
"""Stream WildChat-4.8M shard by shard, keep a hashed sample of English, non-o1 conversations.

Output (gz jsonl, one closed file per shard): work/wc/<shard>.single.jsonl.gz  {id, hash, model, turn, text}           first user turn
                   work/wc/<shard>.multi.jsonl.gz   {id, hash, model, turn, messages}       full convo (2..6 turns)
Each shard is deleted after reading.  Run: python3 -I scripts/wc_extract.py
"""
import gzip, hashlib, json, os, sys
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

REV = "c827c6df8fcf008219ffaffa4d1dd77491099367"
H = os.environ.get("GENERAL_PROMPTS_WORK", "/home/ubuntu/cc-data/general-prompts")
SINGLE_KEEP = 30   # percent of English conversations whose first turn is kept
MULTI_KEEP = 12    # percent of English multi-turn conversations kept whole
COLS = ["conversation_hash", "model", "turn", "language", "toxic", "redacted",
        "conversation.list.element.role", "conversation.list.element.content",
        "conversation.list.element.language"]

def bucket(s):
    return int(hashlib.sha256(("wc:" + s).encode()).hexdigest()[:8], 16) % 100

done = set()
statf = f"{H}/work/wc_stats.jsonl"
if os.path.exists(statf):
    done = {json.loads(l)["shard"] for l in open(statf)}
os.makedirs(f"{H}/work/wc", exist_ok=True)
for i in range(86):
    name = f"data/train-{i:05d}-of-00086.parquet"
    if name in done:
        continue
    tag = f"{i:05d}"
    fs = gzip.open(f"{H}/work/wc/{tag}.single.jsonl.gz.tmp", "wt")
    fm = gzip.open(f"{H}/work/wc/{tag}.multi.jsonl.gz.tmp", "wt")
    p = hf_hub_download("allenai/WildChat-4.8M", name, repo_type="dataset", revision=REV,
                        local_dir=f"{H}/raw/wc")
    st = {"shard": name, "rows": 0, "english": 0, "o1": 0, "redacted": 0, "mixed_lang": 0,
          "single": 0, "multi": 0}
    pf = pq.ParquetFile(p)
    for batch in pf.iter_batches(batch_size=2000, columns=COLS):
        for r in batch.to_pylist():
            st["rows"] += 1
            if r["language"] != "English":
                continue
            st["english"] += 1
            if r["model"].startswith("o1"):
                st["o1"] += 1; continue
            if r["redacted"] or r["toxic"]:
                st["redacted"] += 1; continue
            conv = r["conversation"]
            if not conv or conv[0]["role"] != "user":
                continue
            if any(m["role"] == "user" and m["language"] != "English" for m in conv):
                st["mixed_lang"] += 1; continue
            b = bucket(r["conversation_hash"])
            rid = f"{r['conversation_hash']}"
            if b < SINGLE_KEEP:
                fs.write(json.dumps({"id": rid, "model": r["model"], "turn": r["turn"],
                                     "text": conv[0]["content"]}) + "\n")
                st["single"] += 1
            if r["turn"] >= 2 and (b >= 50 and b < 50 + MULTI_KEEP):
                msgs = [{"role": m["role"], "content": m["content"]} for m in conv]
                if sum(len(m["content"]) for m in msgs) <= 40000:
                    fm.write(json.dumps({"id": rid, "model": r["model"], "turn": r["turn"],
                                         "messages": msgs}) + "\n")
                    st["multi"] += 1
    fs.close(); fm.close()
    os.replace(f"{H}/work/wc/{tag}.single.jsonl.gz.tmp", f"{H}/work/wc/{tag}.single.jsonl.gz")
    os.replace(f"{H}/work/wc/{tag}.multi.jsonl.gz.tmp", f"{H}/work/wc/{tag}.multi.jsonl.gz")
    os.remove(p)
    with open(statf, "a") as f:
        f.write(json.dumps(st) + "\n")
    print(json.dumps(st), flush=True)
print("DONE")
