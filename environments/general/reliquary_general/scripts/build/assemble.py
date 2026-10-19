"""Assemble the corpus: pick rows per block, generate the home-made prompts,
assign thinking modes, split, and write one file per block.

    OMP_NUM_THREADS=1 python -I scripts/assemble.py

Inputs: work/pool_*.jsonl (pools.py). Outputs: final/<block>.jsonl (one row per
(prompt, mode), ordered direct-then-thinking, each run shuffled by key) and
final/assemble_stats.json. Every choice is a hash of the prompt's key, so the
same pools give the same corpus.
"""
from __future__ import annotations

import heapq
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))  # the package root
import gen  # noqa: E402
from common import H, bucket, jdump, norm_key, user_text  # noqa: E402
from reliquary_general import grading  # noqa: E402

W = f"{H}/work"
F = f"{H}/final"
os.makedirs(F, exist_ok=True)

BLOCKS = ("chat", "multiturn", "ifeval", "structured", "safety", "clarification", "identity",
          "tools_call", "tools_pivot")
# (direct, thinking, both) shares, percent.
MODE_SHARES = {
    "chat": (50, 30, 20), "multiturn": (50, 30, 20), "safety": (50, 30, 20), "clarification": (50, 30, 20),
    "identity": (50, 30, 20), "ifeval": (30, 70, 0), "structured": (30, 70, 0), "tools_call": (40, 60, 0),
    "tools_pivot": (40, 60, 0),
}
TOOLS_CALL_PROMPTS = 7_000
TOUCAN_IRRELEVANT = 1_000
MAX_CHARS = {"chat": 12_000, "clarification": 12_000, "multiturn": 16_000, "structured": 24_000, "tools_call": 24_000, "tools_pivot": 32_000}

stats: dict = defaultdict(Counter)


def lines(path):
    with open(path) as f:
        for i, line in enumerate(f):
            yield i, json.loads(line)


def size(r) -> int:
    return len(jdump(r["messages"])) + (len(jdump(r["tools"])) if r.get("tools") else 0)


def select(path, n, salt, pred=lambda r: True, block=None):
    """The n records with the smallest hash among those `pred` accepts."""
    heap: list[tuple[int, int]] = []
    for i, r in lines(path):
        if not pred(r):
            continue
        if block in MAX_CHARS and size(r) > MAX_CHARS[block]:
            stats[block]["over_size_cap"] += 1
            continue
        h = -bucket(r["key"], salt, 1 << 62)
        if len(heap) < n:
            heapq.heappush(heap, (h, i))
        elif h > heap[0][0]:
            heapq.heapreplace(heap, (h, i))
    keep = {i for _, i in heap}
    return [r for i, r in lines(path) if i in keep]


# ----------------------------------------------------------------- WildChat roles


def wildchat_roles():
    """Clarification claims its matches first; the rest splits 70/30 chat / IF base."""
    clar, chat_ids, if_ids = [], [], []
    for i, r in lines(f"{W}/pool_wildchat_single.jsonl"):
        text = r["messages"][0]["content"]
        c = gen.clarification(text) if size(r) <= MAX_CHARS["clarification"] else None
        if c is not None:
            clar.append((bucket(r["key"], "clar", 1 << 62), i, c))
            continue
        if bucket(r["key"], "wc-role", 100) < 70:
            if size(r) <= MAX_CHARS["chat"]:
                chat_ids.append(i)
            else:
                stats["chat"]["over_size_cap"] += 1
        elif gen.if_base_ok(text):
            if_ids.append(i)
    stats["wildchat_roles"].update({"clarification_matches": len(clar), "chat_candidates": len(chat_ids),
                                    "if_candidates": len(if_ids)})
    return clar, chat_ids, if_ids


def take_lines(path, ids):
    ids = set(ids)
    return [r for i, r in lines(path) if i in ids]


def smallest(ids, n, salt, keyf):
    return heapq.nsmallest(n, ids, key=lambda x: bucket(keyf(x), salt, 1 << 62))


# ------------------------------------------------------------------------- blocks


def row(r, block, *, key=None, messages=None, check=None, source=None, meta=None, tools=None):
    return {"key": key or r["key"], "block": block, "source": source or r["src"],
            "messages": messages if messages is not None else r["messages"],
            "tools": tools if tools is not None else r.get("tools"),
            "check": check if check is not None else r.get("check"),
            "origin": meta if meta is not None else r.get("meta", {})}


def build():
    out: dict[str, list] = defaultdict(list)
    clar, chat_ids, if_ids = wildchat_roles()

    # chat
    wc_single = f"{W}/pool_wildchat_single.jsonl"
    chosen_chat = smallest(chat_ids, 36_000, "chat", lambda i: str(i))
    for r in take_lines(wc_single, chosen_chat):
        out["chat"].append(row(r, "chat"))
    for r in select(f"{W}/pool_oasst2.jsonl", 4_500, "chat", lambda r: r["kind"] == "chat", "chat"):
        out["chat"].append(row(r, "chat"))
    for r in select(f"{W}/pool_helpsteer3.jsonl", 4_500, "chat", lambda r: r["kind"] == "chat", "chat"):
        out["chat"].append(row(r, "chat"))

    # multi-turn (frozen history, the last assistant turn is generated)
    wc_multi = f"{W}/pool_wildchat_multi.jsonl"
    for r in select(wc_multi, 7_000, "mt", lambda r: bucket(r["key"], "wc-mt-role", 100) < 75, "multiturn"):
        out["multiturn"].append(row(r, "multiturn"))
    for r in select(f"{W}/pool_oasst2.jsonl", 2_500, "mt", lambda r: r["kind"] == "multiturn", "multiturn"):
        out["multiturn"].append(row(r, "multiturn"))
    for r in select(f"{W}/pool_helpsteer3.jsonl", 2_000, "mt", lambda r: r["kind"] == "multiturn", "multiturn"):
        out["multiturn"].append(row(r, "multiturn"))
    for r in select(f"{W}/pool_multichallenge.jsonl", 1_500, "mt", block="multiturn"):
        out["multiturn"].append(row(r, "multiturn"))

    # verifiable instructions
    def if_row(r, base_messages, source):
        cs = gen.draw_constraints(r["key"])
        if cs is None:
            stats["ifeval"]["constraint_draw_failed"] += 1
            return None
        last = base_messages[-1]["content"]
        prompt = gen.render_constraints(last, cs, r["key"])
        check = {"type": "ifeval", "constraints": [{"id": c["id"], "kwargs": c["kwargs"]} for c in cs]}
        if not grading.builds(check, prompt):
            stats["ifeval"]["check_does_not_build"] += 1
            return None
        msgs = base_messages[:-1] + [{"role": "user", "content": prompt}]
        return row(r, "ifeval", key=r["key"] + "#if", messages=msgs, check=check, source=source,
                   meta={**r.get("meta", {}), "constraints": len(cs), "generated": "ifeval-constraints-v1"})

    n_single = 0
    for r in take_lines(wc_single, smallest(if_ids, 16_500, "if", lambda i: str(i))):
        x = if_row(r, r["messages"], "wildchat+ifevalg")
        if x and n_single < 16_000:
            out["ifeval"].append(x)
            n_single += 1
    n_multi = 0
    for r in select(wc_multi, 4_300, "if-mt", lambda r: bucket(r["key"], "wc-mt-role", 100) >= 75
                    and gen.if_base_ok(r["messages"][-1]["content"]), "multiturn"):
        x = if_row(r, r["messages"], "wildchat-multiturn+ifevalg")
        if x and n_multi < 4_000:
            out["ifeval"].append(x)
            n_multi += 1
    n_ifm = 0
    for r in select(f"{W}/pool_if_multi.jsonl", 6_500, "ifm"):
        prompt = r["messages"][-1]["content"]
        if not grading.builds(r["check"], prompt):
            stats["ifeval"]["if_multi_not_buildable"] += 1
            continue
        if n_ifm < 6_000:
            out["ifeval"].append(row(r, "ifeval"))
            n_ifm += 1

    # structured outputs
    for src, n in (("so_v2", 9_000), ("so_v1", 2_000)):
        for r in select(f"{W}/pool_{src}.jsonl", n, "so", block="structured"):
            if grading.builds(r["check"], ""):
                out["structured"].append(row(r, "structured"))
            else:
                stats["structured"]["schema_does_not_build"] += 1

    # safety: Nemotron Safety prompts, and Aegis balanced between safe and unsafe
    for r in select(f"{W}/pool_safety_v1.jsonl", 9_000, "safety"):
        out["safety"].append(row(r, "safety"))
    for label in ("safe", "unsafe"):
        for r in select(f"{W}/pool_aegis2.jsonl", 1_000, "safety",
                        lambda r, label=label: r["meta"]["prompt_label"] == label):
            out["safety"].append(row(r, "safety"))

    # clarification pairs
    clar.sort()
    chosen = clar[:4_000]
    idx = {i: c for _, i, c in chosen}
    for i, r in lines(wc_single):
        if i not in idx:
            continue
        amputated, slot = idx[i]
        full = r["messages"][0]["content"]
        base = {"pair": r["key"], "slot": slot, "generated": "clarification-pairs-v1"}
        out["clarification"].append(row(r, "clarification", key=r["key"] + "#cut",
                                        messages=[{"role": "user", "content": amputated}],
                                        check={"type": "clarify", "expect": "ask", "slot": slot},
                                        source="wildchat+clarification", meta={**base, "variant": "cut"}))
        out["clarification"].append(row(r, "clarification", key=r["key"] + "#full",
                                        messages=[{"role": "user", "content": full}],
                                        check={"type": "clarify", "expect": "answer", "slot": slot},
                                        source="wildchat+clarification", meta={**base, "variant": "full"}))

    # identity
    seen = set()
    for src in ("identity_nv", "identity_olmo"):
        for _, r in lines(f"{W}/pool_{src}.jsonl"):
            seen.add(norm_key(r["messages"][0]["content"]))
            out["identity"].append(row(r, "identity"))
    added = 0
    for q in gen.identity_paraphrases(5_000):
        if added >= 800:
            break
        k = norm_key(q)
        if k in seen:
            continue
        seen.add(k)
        out["identity"].append({"key": f"identity_gen:{k}", "block": "identity", "source": "identity-templates",
                                "messages": [{"role": "user", "content": q}], "tools": None,
                                "check": {"type": "identity"}, "origin": {"generated": "identity-paraphrases-v1"}})
        added += 1

    # tools. xLAM loses every row whose tools share a name with When2Call (its
    # API pool is xLAM's), about 200 survive; Toucan's single-turn subsets fill
    # the block back to 7,000, three fifths original, two fifths diversified.
    for r in select(f"{W}/pool_xlam.jsonl", 4_000, "tools", block="tools_call"):
        out["tools_call"].append(row(r, "tools_call"))
    fill = TOOLS_CALL_PROMPTS - TOUCAN_IRRELEVANT - len(out["tools_call"])
    original = fill * 3 // 5
    for sub, n in (("single-turn-original", original), ("single-turn-diversify", fill - original),
                   ("irrelevant", TOUCAN_IRRELEVANT)):
        for r in select(f"{W}/pool_toucan.jsonl", n, "tools", lambda r, sub=sub: r["meta"]["subset"] == sub,
                        "tools_call"):
            out["tools_call"].append(row(r, "tools_call"))
    for exp, n in (("function_call", 1_400), ("message", 600)):
        for r in select(f"{W}/pool_fc_pivot.jsonl", n, "tools", lambda r, exp=exp: r["meta"]["expected"] == exp,
                        "tools_pivot"):
            out["tools_pivot"].append(row(r, "tools_pivot"))
    for exp, n in (("function_call", 1_400), ("message", 600)):
        for r in select(f"{W}/pool_conv_pivot.jsonl", n, "tools", lambda r, exp=exp: r["meta"]["expected"] == exp,
                        "tools_pivot"):
            out["tools_pivot"].append(row(r, "tools_pivot"))
    for r in select(f"{W}/pool_toucan.jsonl", 1_500, "tools", lambda r: r["meta"]["subset"] == "multi-turn",
                    "tools_pivot"):
        out["tools_pivot"].append(row(r, "tools_pivot"))
    return out


# ----------------------------------------------------------------- modes, splits


def split_of(base_key: str) -> str:
    b = bucket(base_key, "reliquary_general_v1:split", 100)
    return "train" if b < 96 else "eval" if b < 98 else "qualification"


def modes_of(block: str, mode_key: str) -> tuple[str, ...]:
    direct, thinking, _both = MODE_SHARES[block]
    b = bucket(mode_key, f"mode:{block}", 100)
    if b < direct:
        return ("direct",)
    if b < direct + thinking:
        return ("thinking",)
    return ("direct", "thinking")


def single_turn(r) -> bool:
    """One user turn, no tools: no system message, no history, no transcript."""
    return len(r["messages"]) == 1 and r["messages"][0]["role"] == "user" and not r.get("tools")


def write(out):
    summary = {}
    for block in BLOCKS:
        rows = out[block]
        keys = [r["key"] for r in rows]
        assert len(keys) == len(set(keys)), f"duplicate keys in {block}"
        runs = {"direct": [], "thinking": []}
        for r in rows:
            # A clarification pair shares its split and its modes, so each half of
            # the pair lands beside the other.
            base_key = r["origin"].get("pair") or r["key"]
            single = single_turn(r)
            for mode in modes_of(block, base_key):
                runs[mode].append({"split": split_of(base_key), "mode": mode, "single_turn": single,
                                   "key": r["key"], **{
                    k: r[k] for k in ("block", "source", "messages", "tools", "check", "origin")}})
        with open(f"{F}/{block}.jsonl", "w") as f:
            for mode in ("direct", "thinking"):
                # Single-turn rows first, so that the part of a segment a plain
                # one-turn renderer can serve is one contiguous range.
                for x in sorted(runs[mode], key=lambda x: (not x["single_turn"],
                                                           bucket(x["key"], f"order:{block}:{mode}", 1 << 62))):
                    f.write(json.dumps(x, ensure_ascii=False, separators=(",", ":")) + "\n")
        summary[block] = {
            "prompts": len(rows),
            "by_source": dict(Counter(r["source"] for r in rows)),
            "rows": {m: dict(Counter(x["split"] for x in runs[m])) for m in runs},
            "graded": sum(r["check"] is not None for r in rows),
            "single_turn_rows": {m: dict(Counter(x["split"] for x in runs[m] if x["single_turn"])) for m in runs},
            "bytes": os.path.getsize(f"{F}/{block}.jsonl"),
        }
        print(block, json.dumps(summary[block]), flush=True)
    json.dump({"blocks": summary, "stats": {k: dict(v) for k, v in stats.items()}},
              open(f"{F}/assemble_stats.json", "w"), indent=1)


if __name__ == "__main__":
    write(build())
