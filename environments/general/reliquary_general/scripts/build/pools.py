"""Filter, deduplicate and decontaminate every source into work/pool_<src>.jsonl.

    OMP_NUM_THREADS=1 python -I scripts/pools.py

Streaming, in priority order: the first source to claim a prompt keeps it, so a
WildChat prompt that HelpSteer3 already carries is dropped from WildChat, and
any prompt near-identical to one in the instruction-following environment's
corpus is dropped everywhere. Stats land in work/pool_stats.json.
"""
from __future__ import annotations

import gzip
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from common import (H, MIDJOURNEY, NearDup, bucket, extreme, has_pii, jdump, junk, norm_key, user_text, wc)  # noqa: E402
from decon import EvalIndex, norm_fn  # noqa: E402

import langid  # noqa: E402

W = f"{H}/work"
IF_ENV = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "instruction_following",
                      "reliquary_instruction_following", "reliquary_instruction_following", "data",
                      "prompts.jsonl.gz")

stats: dict = defaultdict(Counter)
decon_hits: dict = defaultdict(Counter)

evals = EvalIndex()
text_dedup = NearDup(0.7)
exact_seen: set[str] = set()
long_dedup = NearDup(0.8)  # structured documents and tool prompts


def seed_if_env():
    """The IF environment's 37k prompts (WildChat-1M) are claimed up front, so
    nothing here repeats a prompt Teutonic already trains on there."""
    n = 0
    for line in gzip.open(IF_ENV, "rt"):
        p = json.loads(line)["prompt"]
        base = p.split(" Please follow these rules when answering:")[0]
        exact_seen.add(norm_key(base))
        text_dedup.seen(base)
        n += 1
    stats["if_env_seed"]["prompts"] = n


def english(text: str) -> bool:
    sample = text[:2000]
    return langid.classify(sample)[0] == "en"


def contaminated(src: str, text: str) -> bool:
    hits = evals.hits(text)
    if hits:
        for bench in sorted({b for b, _ in hits}):
            decon_hits[src][bench] += 1
        return True
    return False


def contaminated_docs(src: str, docs: str) -> bool:
    """Tool descriptions against eval tool docs, When2Call excepted: When2Call
    reuses xLAM's RapidAPI pool, so its descriptions are xLAM's by construction.
    Its queries are still matched (a quarter of xLAM's queries reappear there)."""
    hits = [h for h in evals.hits(docs) if h[0] != "when2call"]
    for bench in sorted({b for b, _ in hits}):
        decon_hits[src][bench + ":tool_doc"] += 1
    return bool(hits)


def keep_text(src: str, text: str, *, min_words=3, max_words=2000, check_lang=True,
              allow_extreme=False, near=True) -> bool:
    st = stats[src]
    st["in"] += 1
    if not text.strip():
        st["empty"] += 1
        return False
    n = wc(text)
    if n < min_words or n > max_words:
        st["length"] += 1
        return False
    if junk(text):
        st["junk"] += 1
        return False
    if has_pii(text):
        st["pii"] += 1
        return False
    if not allow_extreme and extreme(text):
        st["toxic"] += 1
        return False
    if check_lang and not english(text):
        st["not_english"] += 1
        return False
    k = norm_key(text)
    if k in exact_seen:
        st["dup_exact"] += 1
        return False
    if near and text_dedup.seen(text):
        st["dup_near"] += 1
        exact_seen.add(k)
        return False
    exact_seen.add(k)
    if contaminated(src, text):
        st["decontaminated"] += 1
        return False
    st["kept"] += 1
    return True


def wc_records(kind):
    import glob
    for path in sorted(glob.glob(f"{W}/wc/*.{kind}.jsonl.gz")):
        yield from records(path)


def records(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        for line in f:
            yield json.loads(line)


def write(src, it):
    with open(f"{W}/pool_{src}.jsonl", "w") as f:
        for r in it:
            f.write(jdump(r) + "\n")


# ------------------------------------------------------------------ simple kinds


def simple(src, file, **kw):
    def gen():
        for r in records(f"{W}/src_{file}.jsonl"):
            if r["src"] != src:
                continue
            text = r.get("base") or user_text(r["messages"])
            if kw.get("max_chars") and sum(len(m["content"]) for m in r["messages"]) > kw["max_chars"]:
                stats[src]["in"] += 1
                stats[src]["context_too_long"] += 1
                continue
            if keep_text(src, text, **{k: v for k, v in kw.items() if k != "max_chars"}):
                yield r
    write(src, gen())


def wildchat_multi():
    src = "wildchat_multi"

    def gen():
        for r in wc_records("multi"):
            msgs = r["messages"]
            # Ends on a user turn: drop the last assistant reply; the target is regenerated.
            while msgs and msgs[-1]["role"] != "user":
                msgs = msgs[:-1]
            if len(msgs) < 3 or any(m["role"] not in ("user", "assistant") for m in msgs):
                stats[src]["in"] += 1
                stats[src]["shape"] += 1
                continue
            if any(not m["content"].strip() for m in msgs):
                stats[src]["in"] += 1
                stats[src]["empty_turn"] += 1
                continue
            if sum(len(m["content"]) for m in msgs) > 24_000:
                stats[src]["in"] += 1
                stats[src]["context_too_long"] += 1
                continue
            if any(has_pii(m["content"]) or extreme(m["content"]) for m in msgs if m["role"] == "assistant"):
                stats[src]["in"] += 1
                stats[src]["assistant_pii_or_toxic"] += 1
                continue
            if any(junk(m["content"]) for m in msgs if m["role"] == "user"):
                # "continue"/"thanks" turns make poor targets and poor context.
                stats[src]["in"] += 1
                stats[src]["junk_turn"] += 1
                continue
            if keep_text(src, user_text(msgs), check_lang=False, max_words=4000):
                yield {"key": f"wildchat:{r['id']}", "src": "wildchat", "kind": "multiturn", "messages": msgs,
                       "tools": None, "check": None,
                       "meta": {"model": r["model"], "turns": sum(m["role"] == "user" for m in msgs)}}
    write(src, gen())


def wildchat_single():
    """~45% of the extracted English first turns, by hash, to bound memory."""
    src = "wildchat_single"

    def gen():
        for r in wc_records("single"):
            if bucket(r["id"], "wc-presample", 100) >= 45:
                continue
            text = r["text"]
            if keep_text(src, text, check_lang=False):
                yield {"key": f"wildchat:{r['id']}", "src": "wildchat", "kind": "chat",
                       "messages": [{"role": "user", "content": text}], "tools": None, "check": None,
                       "meta": {"model": r["model"], "turn": r["turn"]}}
    write(src, gen())


# ------------------------------------------------------------- long-form kinds


def long_kind(src, file, *, max_chars):
    def gen():
        for r in records(f"{W}/src_{file}.jsonl"):
            if r["src"] != src:
                continue
            st = stats[src]
            st["in"] += 1
            body = jdump(r["messages"])
            if len(body) > max_chars:
                st["context_too_long"] += 1
                continue
            text = user_text(r["messages"])
            if has_pii(text):
                st["pii"] += 1
                continue
            k = norm_key(body + jdump(r.get("check")))
            if k in exact_seen:
                st["dup_exact"] += 1
                continue
            exact_seen.add(k)
            if long_dedup.seen(text):
                st["dup_near"] += 1
                continue
            if contaminated(src, text):
                st["decontaminated"] += 1
                continue
            st["kept"] += 1
            yield r
    write(src, gen())


def tool_kind(src, file, *, max_chars=48_000, sample_mod=None, name_benches=("tau2",)):
    """Tool prompts: also refuse a row whose tools carry a function name of the
    benches in `name_benches`, and any tool description sharing 8-grams with an
    eval's tool docs. Names are matched against tau2 everywhere (its domain
    tools); against BFCL and When2Call only for xLAM: BFCL for its 134
    BFCL-inspired functions, When2Call because it is built from xLAM's own
    RapidAPI pool (3,119 of the 3,390 normalised function names left after the
    BFCL pass), so an xLAM row calling one of them rehearses a When2Call item's
    tools even when its query differs. That drops all but ~200 xLAM rows; the
    tool-call block is refilled from Toucan. Generic names (get_weather,
    send_email, add) recur across BFCL and When2Call by design; matching them
    in the other sources would drop most of the tool block for no
    contamination of the queries, which the 8-gram pass covers, so there the
    overlap is recorded, not filtered."""
    def gen():
        for r in records(f"{W}/src_{file}.jsonl"):
            if sample_mod and bucket(r["key"], f"{src}-presample", 100) >= sample_mod:
                continue
            st = stats[src]
            st["in"] += 1
            body = jdump(r["messages"]) + jdump(r["tools"])
            if len(body) > max_chars:
                st["context_too_long"] += 1
                continue
            text = "\n".join(m["content"] for m in r["messages"] if m["role"] in ("user", "system"))
            if has_pii(text):
                st["pii"] += 1
                continue
            names = {norm_fn(t["name"]) for t in r["tools"]}
            for bench, fns in evals.function_names.items():
                if names & fns:
                    decon_hits[src][bench + ":function_name_overlap"] += 1
            if any(names & evals.function_names[b] for b in name_benches):
                st["eval_function_name"] += 1
                continue
            k = norm_key(body + jdump(r.get("check")))
            if k in exact_seen:
                st["dup_exact"] += 1
                continue
            exact_seen.add(k)
            docs = "\n".join(t.get("description") or "" for t in r["tools"])
            if contaminated(src, text) or contaminated_docs(src, docs):
                st["decontaminated"] += 1
                continue
            st["kept"] += 1
            yield r
    write(src, gen())


def main():
    seed_if_env()
    print("seeded", dict(stats["if_env_seed"]), flush=True)
    # Curated sources first: they claim their prompts before WildChat does.
    simple("identity_nv", "identity", min_words=1, check_lang=False)
    simple("identity_olmo", "identity", min_words=1, check_lang=False)
    simple("if_multi", "if_multi", max_words=600)
    simple("safety_v1", "safety", min_words=2, allow_extreme=True, max_words=1500)
    simple("aegis2", "safety", min_words=2, allow_extreme=True, max_words=1500)
    simple("oasst2", "oasst2", max_words=4000, max_chars=24_000)
    simple("multichallenge", "multichallenge", max_words=8000, max_chars=48_000, check_lang=False)
    simple("helpsteer3", "helpsteer3", max_words=4000, max_chars=24_000)
    print("curated done", flush=True)
    wildchat_multi()
    print("wc multi done", flush=True)
    wildchat_single()
    print("wc single done", flush=True)
    long_kind("so_v2", "structured", max_chars=40_000)
    long_kind("so_v1", "structured", max_chars=40_000)
    tool_kind("xlam", "xlam", name_benches=("tau2", "bfcl_v4", "when2call"))
    tool_kind("fc_pivot", "fc_pivot")
    tool_kind("conv_pivot", "conv_pivot")
    tool_kind("toucan", "toucan", sample_mod=25)
    out = {"stats": {k: dict(v) for k, v in stats.items()},
           "decontamination": {k: dict(v) for k, v in decon_hits.items()},
           "eval_boilerplate_grams": evals.boilerplate}
    json.dump(out, open(f"{W}/pool_stats.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
