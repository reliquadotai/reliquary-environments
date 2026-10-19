import os
"""Download eval benchmarks and normalise them to evals/<bench>.jsonl for 8-gram decontamination.

Rerunnable: raw files are cached in evals/raw/<bench>/ and skipped if present.
Run with: .venv/bin/python -I scripts/evals_dl.py
"""
import ast, csv, glob, io, json, os, re, subprocess, sys, urllib.request
from collections import Counter

BASE = os.environ.get("GENERAL_PROMPTS_WORK", "/home/ubuntu/cc-data/general-prompts") + "/evals"
RAW = os.path.join(BASE, "raw")
TOKEN_FILE = os.path.expanduser("~/.config/reliquary/hf-subnet.token")

HF = {  # bench -> (repo, sha, files)
    "ifeval": ("google/IFEval", "966cd89545d6b6acfd7638bc708b98261ca58e84", ["ifeval_input_data.jsonl"]),
    "ifbench": ("allenai/IFBench_test", "2e8a48de45ff3bf41242f927254ca81b59ca3ae2", ["data/train-00000-of-00001.parquet"]),
    "arena_hard_v2": ("lmarena-ai/arena-hard-auto", "15f3746e21432264ce9b453999bde4f3c946d2e6", ["data/arena-hard-v2.0/question.jsonl"]),
    "mt_bench": ("HuggingFaceH4/mt_bench_prompts", "e3a795c5e9a82ee40611c416b8a7786c73198991", ["raw/question.jsonl"]),
    "when2call": ("nvidia/When2Call", "0582f7749df63a96fdc3070932e83e72396ace53", [
        "test/when2call_test_llm_judge.jsonl", "test/when2call_test_mcq.jsonl",
        "train/when2call_train_pref.jsonl", "train/when2call_train_sft.jsonl"]),
    "xstest": ("natolambert/xstest-v2-copy", "b71afe2a6d10e5a6254ea8bcb006c48b095a15d5", ["data/prompts-00000-of-00001.parquet"]),
    "mmlu_pro": ("TIGER-Lab/MMLU-Pro", "b189ec765aa7ed75c8acfea42df31fdae71f97be", ["data/test-00000-of-00001.parquet"]),
    "gpqa_diamond": ("Idavidrein/gpqa", "83022cefff930aea54f654c0b282e74b9eeda5c6", ["gpqa_diamond.csv"]),
}
GORILLA = ("ShishirPatil/gorilla", "6ea57973c7a6097fd7c5915698c54c17c5b1b6c8")
TAU2 = ("sierra-research/tau2-bench", "4ce7c0397c1eb65c9bbe59aeacfe1ca44a1cd699")
BFCL_PREFIX = "berkeley-function-call-leaderboard/bfcl_eval/data/"
TAU2_FILES = [
    "data/tau2/domains/airline/tasks.json", "data/tau2/domains/airline/policy.md",
    "data/tau2/domains/retail/tasks.json", "data/tau2/domains/retail/policy.md",
    "data/tau2/domains/telecom/tasks.json", "data/tau2/domains/telecom/main_policy.md",
    "data/tau2/domains/telecom/main_policy_solo.md", "data/tau2/domains/telecom/tech_support_manual.md",
    "data/tau2/domains/telecom/tech_support_workflow.md", "data/tau2/domains/telecom/tech_support_workflow_solo.md",
    "src/tau2/domains/airline/tools.py", "src/tau2/domains/retail/tools.py",
    "src/tau2/domains/telecom/tools.py", "src/tau2/domains/telecom/user_tools.py",
]
LICENCES = {}


def hf_token():
    with open(TOKEN_FILE) as f:
        return "".join(f.read().split())


def dl_hf(bench):
    from huggingface_hub import hf_hub_download
    repo, sha, files = HF[bench]
    out = os.path.join(RAW, bench)
    os.makedirs(out, exist_ok=True)
    tok = hf_token() if bench == "gpqa_diamond" else None
    for f in files:
        if not os.path.exists(os.path.join(out, f)):
            hf_hub_download(repo, f, repo_type="dataset", revision=sha, local_dir=out, token=tok)
    return [os.path.join(out, f) for f in files]


def gh_tree(repo, sha):
    p = subprocess.run(["gh", "api", f"repos/{repo}/git/trees/{sha}?recursive=1"], capture_output=True, text=True, check=True)
    return json.loads(p.stdout)["tree"]


def dl_gh(bench, repo, sha, paths):
    out = os.path.join(RAW, bench)
    for p in paths:
        dst = os.path.join(out, p)
        if os.path.exists(dst):
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        url = f"https://raw.githubusercontent.com/{repo}/{sha}/{p}"
        subprocess.run(["curl", "-sfL", "--retry", "3", "-o", dst, url], check=True)
    return [os.path.join(out, p) for p in paths]


def download():
    for b in HF:
        try:
            dl_hf(b)
        except Exception as e:
            print(f"[{b}] HF download FAILED: {type(e).__name__}: {str(e)[:200]}", file=sys.stderr)
    bf = os.path.join(RAW, "bfcl_v4", "_paths.json")
    if not os.path.exists(bf):
        tree = gh_tree(*GORILLA)
        paths = [t["path"] for t in tree if t["type"] == "blob" and t["path"].startswith(BFCL_PREFIX)
                 and t["path"].endswith(".json") and "/possible_answer/" not in t["path"]
                 and "executable_eval_sanity_check" not in t["path"]]
        os.makedirs(os.path.dirname(bf), exist_ok=True)
        json.dump(paths, open(bf, "w"))
    dl_gh("bfcl_v4", *GORILLA, json.load(open(bf)))
    dl_gh("tau2", *TAU2, TAU2_FILES)




LICENCES = {
    "ifeval": "apache-2.0 (HF card)", "ifbench": "ODC-BY-1.0 (HF README)", "arena_hard_v2": "apache-2.0 (HF card)",
    "mt_bench": "apache-2.0 (HF card)", "when2call": "cc-by-4.0 (HF card)", "xstest": "cc-by-4.0 (HF card)",
    "mmlu_pro": "mit (HF card)", "gpqa_diamond": "cc-by-4.0 (HF card; gated, no-web-reposting request)",
    "bfcl_v4": "Apache-2.0 (GitHub repo licence)", "tau2": "MIT (GitHub repo licence)",
}


class Out:
    """Collects rows for one bench; drops exact (kind, text) duplicates."""

    def __init__(self, bench):
        self.bench, self.rows, self.seen, self.dups = bench, [], set(), 0

    def add(self, id_, text, kind="prompt"):
        if not isinstance(text, str):
            return
        text = text.strip()
        if not text:
            return
        k = (kind, text)
        if k in self.seen:
            self.dups += 1
            return
        self.seen.add(k)
        self.rows.append({"bench": self.bench, "id": str(id_), "text": text, "kind": kind})

    def write(self):
        with open(os.path.join(BASE, self.bench + ".jsonl"), "w") as f:
            for r in self.rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return dict(Counter(r["kind"] for r in self.rows)), self.dups


def jsonl(path):
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def rel(paths):
    return [os.path.relpath(p, os.path.join(RAW)) for p in paths]


def b_ifeval(o):
    (p,) = dl_hf("ifeval")
    for d in jsonl(p):
        o.add(d["key"], d["prompt"])
    return [HF["ifeval"][2][0]], {}


def b_ifbench(o):
    import pyarrow.parquet as pq
    (p,) = dl_hf("ifbench")
    t = pq.read_table(p, columns=["key", "prompt", "instruction_id_list"]).to_pylist()
    ids = set()
    for d in t:
        o.add(d["key"], d["prompt"])
        ids.update(d["instruction_id_list"])
    ids = sorted(ids)
    json.dump(ids, open(os.path.join(BASE, "ifbench_constraint_ids.json"), "w"), indent=1)
    return HF["ifbench"][2], {"n_rows": len(t), "n_constraint_ids": len(ids)}


def b_arena(o):
    (p,) = dl_hf("arena_hard_v2")
    cats = Counter()
    for d in jsonl(p):
        o.add(d["uid"], d["prompt"])
        cats[d.get("category")] += 1
    return HF["arena_hard_v2"][2], {"categories": dict(cats)}


def b_mtbench(o):
    (p,) = dl_hf("mt_bench")
    n = 0
    for d in jsonl(p):
        n += 1
        turns = d.get("turns") or d.get("prompt")
        for i, turn in enumerate(turns):
            o.add(f"{d['question_id']}/turn{i + 1}", turn)
    return HF["mt_bench"][2], {"n_questions": n}


def add_tools(o, prefix, funcs):
    for fn in funcs:
        if isinstance(fn, str):
            fn = json.loads(fn)
        name = fn.get("name")
        o.add(f"{prefix}/{name}", name, "function_name")
        o.add(f"{prefix}/{name}", fn.get("description"), "tool_doc")


def b_when2call(o):
    paths = dl_hf("when2call")
    per = {}
    for p in paths:
        split = os.path.basename(p).replace(".jsonl", "")
        n = 0
        for i, d in enumerate(jsonl(p)):
            n += 1
            rid = d.get("uuid") or f"{split}/{i}"
            if "question" in d:
                o.add(rid, d["question"])
            for j, m in enumerate(d.get("messages") or []):
                if m.get("role") == "user":
                    o.add(f"{rid}/u{j}", m["content"])
            add_tools(o, rid, d.get("tools") or [])
            add_tools(o, rid, d.get("orig_tools") or [])
        per[split] = n
    return HF["when2call"][2], {"rows_per_file": per}


def b_xstest(o):
    import pyarrow.parquet as pq
    (p,) = dl_hf("xstest")
    t = pq.read_table(p, columns=["id", "type", "prompt"]).to_pylist()
    c = Counter("unsafe" if d["type"].startswith("contrast") else "safe" for d in t)
    for d in t:
        o.add(d["id"], d["prompt"])
    return HF["xstest"][2], {"safe_unsafe": dict(c)}


def b_mmlu_pro(o):
    import pyarrow.parquet as pq
    (p,) = dl_hf("mmlu_pro")
    t = pq.read_table(p, columns=["question_id", "question"])
    for qid, q in zip(t.column("question_id").to_pylist(), t.column("question").to_pylist()):
        o.add(qid, q)
    return HF["mmlu_pro"][2], {"n_rows": t.num_rows, "columns_read": ["question_id", "question"]}


def b_gpqa(o):
    local = "/home/ubuntu/cc-data/gpqa-check/gpqa_extended.csv"
    with open(local, newline="") as f:
        cols = next(csv.reader(f))
    has_diamond = any("diamond" in c.lower() for c in cols)
    (p,) = dl_hf("gpqa_diamond")
    with open(p, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        o.add(r["Record ID"], r["Question"])
    return HF["gpqa_diamond"][2], {"n_rows": len(rows), "local_extended_has_diamond_column": has_diamond}


MT_DOC_CLASS = {
    "gorilla_file_system": "GorillaFileSystem", "math_api": "MathAPI", "message_api": "MessageAPI",
    "posting_api": "TwitterAPI", "ticket_api": "TicketAPI", "trading_bot": "TradingBot",
    "travel_booking": "TravelAPI", "vehicle_control": "VehicleControlAPI", "web_search": "WebSearchAPI",
    "memory_kv": "MemoryAPI_kv", "memory_vector": "MemoryAPI_vector", "memory_rec_sum": "MemoryAPI_rec_sum",
}


def bfcl_turns(q):
    """question is [[msg,...], ...] (multi-turn) or [msg,...] (single)."""
    if isinstance(q, str):
        q = [[{"role": "user", "content": q}]]
    if q and isinstance(q[0], dict):
        q = [q]
    for ti, turn in enumerate(q or []):
        for mi, m in enumerate(turn or []):
            if isinstance(m, dict):
                yield ti, mi, m


def b_bfcl(o):
    paths = dl_gh("bfcl_v4", *GORILLA, json.load(open(os.path.join(RAW, "bfcl_v4", "_paths.json"))))
    per = {}
    for p in sorted(paths):
        rp = p.split(BFCL_PREFIX, 1)[1]
        stem = rp.replace(".json", "")
        if "multi_turn_func_doc/" in rp:
            cls = MT_DOC_CLASS.get(os.path.basename(stem), os.path.basename(stem))
            n = 0
            for fn in jsonl(p):
                n += 1
                o.add(f"{cls}.{fn['name']}", fn["name"], "function_name")
                o.add(f"{cls}.{fn['name']}", fn.get("description"), "tool_doc")
            per[rp] = n
            continue
        try:
            items = list(jsonl(p))
        except json.JSONDecodeError:
            per[rp] = "skipped (not JSONL: id index)"
            continue
        n = 0
        for i, d in enumerate(items):
            if "question" not in d:
                continue
            d.setdefault("id", f"idx{i}")
            n += 1
            for ti, mi, m in bfcl_turns(d["question"]):
                if m.get("role") in ("user", "system"):
                    o.add(f"{stem}/{d['id']}/t{ti}m{mi}", m.get("content"))
            add_tools(o, f"{stem}/{d['id']}", d.get("function") or [])
        per[rp] = n
    return [BFCL_PREFIX + "<files below>"], {"entries_per_file": per}


def tau2_tools(o, dom, path):
    tree = ast.parse(open(path).read())
    n = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            decs = [ast.unparse(d) for d in node.decorator_list]
            if any(d.startswith("is_tool") for d in decs):
                n += 1
                o.add(f"{dom}/tool/{node.name}", node.name, "function_name")
                o.add(f"{dom}/tool/{node.name}", ast.get_docstring(node), "tool_doc")
    return n


def b_tau2(o):
    paths = dl_gh("tau2", *TAU2, TAU2_FILES)
    info = {}
    for p in paths:
        rp = p.split(os.path.join(RAW, "tau2") + "/", 1)[1]
        dom = rp.split("/")[3]
        if rp.endswith("tasks.json"):
            tasks = json.load(open(p))
            for t in tasks:
                us = t.get("user_scenario") or {}
                ins = us.get("instructions")
                tid = f"{dom}/{t['id']}"
                o.add(f"{tid}/persona", us.get("persona"))
                if isinstance(ins, str):
                    o.add(f"{tid}/instructions", ins)
                elif isinstance(ins, dict):
                    for k in ("reason_for_call", "known_info", "unknown_info", "task_instructions"):
                        o.add(f"{tid}/{k}", ins.get(k))
                tk = (t.get("ticket") if isinstance(t.get("ticket"), str) else None)
                o.add(f"{tid}/ticket", tk)
            info[rp] = len(tasks)
        elif rp.endswith(".md"):
            paras = [x for x in re.split(r"\n\s*\n", open(p).read()) if x.strip()]
            for i, para in enumerate(paras):
                o.add(f"{rp}#p{i}", para, "tool_doc")
            info[rp] = len(paras)
        elif rp.endswith(".py"):
            info[rp] = tau2_tools(o, dom + ("/user" if "user_tools" in rp else ""), p)
    return TAU2_FILES, {"items_per_file": info}


BUILDERS = {
    "ifeval": b_ifeval, "ifbench": b_ifbench, "arena_hard_v2": b_arena, "mt_bench": b_mtbench,
    "bfcl_v4": b_bfcl, "tau2": b_tau2, "when2call": b_when2call, "xstest": b_xstest,
    "mmlu_pro": b_mmlu_pro, "gpqa_diamond": b_gpqa,
}


def source(b):
    if b in HF:
        return {"source": f"https://huggingface.co/datasets/{HF[b][0]}", "revision": HF[b][1]}
    repo, sha = GORILLA if b == "bfcl_v4" else TAU2
    return {"source": f"https://github.com/{repo}", "revision": sha}


def main():
    download()
    man = {}
    only = sys.argv[1:]
    for b, fn in BUILDERS.items():
        if only and b not in only:
            continue
        o = Out(b)
        try:
            files, extra = fn(o)
        except Exception as e:
            print(f"[{b}] FAILED: {type(e).__name__}: {e}", file=sys.stderr)
            man[b] = {**source(b), "error": f"{type(e).__name__}: {e}"}
            continue
        counts, dups = o.write()
        man[b] = {**source(b), "files": files, "lines_per_kind": counts,
                  "exact_duplicates_dropped": dups, "licence": LICENCES[b], **extra}
        print(b, counts, "dups", dups)
    mp = os.path.join(BASE, "MANIFEST.json")
    old = json.load(open(mp)) if (only and os.path.exists(mp)) else {}
    old.update(man)
    json.dump(old, open(mp, "w"), indent=1)
    for p in glob.glob(os.path.join(RAW, "**", "*"), recursive=True):
        if os.path.isfile(p) and os.path.getsize(p) > 50 * 2**20:
            os.remove(p)


if __name__ == "__main__":
    main()
