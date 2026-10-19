"""Normalise every non-WildChat source into one record shape.

    python -I scripts/extract.py <source> [...]

Each source writes work/src_<source>.jsonl, one record per line:

    key       "<source>:<upstream id>"           stable identity
    src       source name
    kind      chat | multiturn | ifeval | structured | safety | identity | tool_call | tool_pivot
    messages  [{role, content, tool_calls?, name?}]   history, ends on a user or tool turn
    tools     [{name, description, parameters}] | null
    check     what the grader needs, or null
    meta      upstream fields worth keeping
    base      text the decontamination reads (defaults to every user turn)

Downloaded files are data only: nothing here executes or imports from them.
"""
from __future__ import annotations

import ast
import gzip
import hashlib
import json
import os
import re
import sys

import pyarrow.parquet as pq

H = os.environ.get("GENERAL_PROMPTS_WORK", "/home/ubuntu/cc-data/general-prompts")
R = f"{H}/raw"


def out(name):
    return open(f"{H}/work/src_{name}.jsonl", "w")


def rec(f, **kw):
    kw.setdefault("tools", None)
    kw.setdefault("check", None)
    kw.setdefault("meta", {})
    f.write(json.dumps(kw, ensure_ascii=False) + "\n")


def merge_same_role(messages):
    """Consecutive user messages become one; the chat template would otherwise
    render two user turns in a row, which no generation server expects."""
    merged = []
    for m in messages:
        if merged and m["role"] == "user" and merged[-1]["role"] == "user":
            merged[-1] = {"role": "user", "content": merged[-1]["content"] + "\n\n" + m["content"]}
        else:
            merged.append(dict(m))
    return merged


# --------------------------------------------------------------------------- chat


def oasst2():
    """Roots as single-turn chat; one prompter-ending path per tree as multi-turn.

    Assistant turns in oasst2 were written by human volunteers, so the frozen
    history carries no model output.
    """
    rows = []
    for fn in ("train-00000-of-00001-88ba0162028a73fc", "validation-00000-of-00001-1deeef95c3248fe0"):
        pf = pq.ParquetFile(f"{R}/OpenAssistant__oasst2/data/{fn}.parquet")
        for b in pf.iter_batches(batch_size=5000, columns=[
                "message_id", "parent_id", "text", "role", "lang", "review_result", "deleted",
                "rank", "synthetic", "message_tree_id"]):
            rows.extend(b.to_pylist())
    by_id = {r["message_id"]: r for r in rows}
    kids = {}
    for r in rows:
        if r["parent_id"]:
            kids.setdefault(r["parent_id"], []).append(r)

    def ok(r):
        return r["lang"] == "en" and not r["deleted"] and r["review_result"] is not False and not r["synthetic"]

    f = out("oasst2")
    n_root = n_multi = 0
    for r in rows:
        if r["parent_id"] is None and r["role"] == "prompter" and ok(r):
            rec(f, key=f"oasst2:{r['message_id']}", src="oasst2", kind="chat",
                messages=[{"role": "user", "content": r["text"]}], meta={"tree": r["message_tree_id"]})
            n_root += 1
            # Deepest path that follows the best-ranked assistant reply and the
            # first reviewed prompter follow-up, ending on a prompter turn.
            path = [r]
            node = r
            while True:
                children = [c for c in kids.get(node["message_id"], []) if ok(c)]
                if not children:
                    break
                if node["role"] == "prompter":
                    children.sort(key=lambda c: (c["rank"] if c["rank"] is not None else 99, c["message_id"]))
                else:
                    children.sort(key=lambda c: c["message_id"])
                node = children[0]
                path.append(node)
                if len(path) >= 7:
                    break
            while path and path[-1]["role"] != "prompter":
                path.pop()
            if len(path) >= 3:
                msgs = [{"role": "user" if m["role"] == "prompter" else "assistant", "content": m["text"]}
                        for m in path]
                rec(f, key=f"oasst2:{path[-1]['message_id']}", src="oasst2", kind="multiturn",
                    messages=msgs, meta={"tree": r["message_tree_id"], "turns": (len(msgs) + 1) // 2})
                n_multi += 1
    print("oasst2 roots", n_root, "multi", n_multi)


def helpsteer3():
    """General and STEM prompts only (curated from WildChat-1M upstream; deduplicated
    against our WildChat draw later). Responses and preferences are discarded."""
    f = out("helpsteer3")
    seen = set()
    n = {"chat": 0, "multiturn": 0}
    for line in gzip.open(f"{R}/nvidia__HelpSteer3/preference/train.jsonl.gz", "rt"):
        r = json.loads(line)
        if r["domain"] not in ("general", "stem"):
            continue
        ctx = r["context"]
        if not ctx or ctx[-1]["role"] != "user" or any(m["role"] not in ("user", "assistant") for m in ctx):
            continue
        msgs = [{"role": m["role"], "content": m["content"]} for m in ctx]
        h = hashlib.sha256(json.dumps(msgs).encode()).hexdigest()[:20]
        if h in seen:
            continue
        seen.add(h)
        kind = "chat" if len(msgs) == 1 else "multiturn"
        rec(f, key=f"helpsteer3:{h}", src="helpsteer3", kind=kind, messages=msgs,
            meta={"domain": r["domain"]})
        n[kind] += 1
    print("helpsteer3", n)


def multichallenge():
    """Hard multi-turn conversations (vanilla: user/assistant; advanced: with a
    role-play system prompt that is part of the task). The upstream rubric is a
    judge, so these are ungraded."""
    f = out("multichallenge")
    n = 0
    for fn in ("vanilla", "advanced"):
        for line in open(f"{R}/nvidia__Nemotron-RL-Multichallenge-v1/data/{fn}.jsonl"):
            r = json.loads(line)
            if r["language"] != "en" or r["messages"][-1]["role"] != "user":
                continue
            msgs = [{"role": m["role"], "content": m["content"]} for m in r["messages"]]
            rec(f, key=f"multichallenge:{r['uuid']}", src="multichallenge", kind="multiturn",
                messages=msgs, meta={"subset": fn, "turns": sum(m["role"] == "user" for m in msgs)})
            n += 1
    print("multichallenge", n)


# -------------------------------------------------------------------- instructions

IF_KEEP_PREFIX = ("wildchat", "oasst1", "flan_v2", "gsm8k", "numinamath")


def if_multi():
    """IF_multi_constraints_upto5, only rows whose base prompt comes from a source
    the report cleared (WildChat, oasst1, FLAN v2, GSM8K, NuminaMath)."""
    f = out("if_multi")
    pf = pq.ParquetFile(f"{R}/allenai__IF_multi_constraints_upto5/data/train-00000-of-00001.parquet")
    n = {}
    for b in pf.iter_batches(batch_size=5000):
        for r in b.to_pylist():
            key = r["key"]
            src = key.split("/")[-1].lower()
            tag = next((p for p in IF_KEEP_PREFIX if p in src), None)
            n[tag] = n.get(tag, 0) + 1
            if tag is None:
                continue
            gt = ast.literal_eval(r["ground_truth"])
            assert len(gt) == 1
            ids, kwargs = gt[0]["instruction_id"], gt[0]["kwargs"]
            constraints = [{"id": i, "kwargs": {k: v for k, v in (kw or {}).items() if v is not None}}
                           for i, kw in zip(ids, kwargs)]
            prompt = r["messages"][0]["content"]
            base = prompt
            for c in r["constraint"].split("\t"):
                base = base.replace(c.strip(), " ")
            rec(f, key=f"if_multi:{key}", src="if_multi", kind="ifeval",
                messages=[{"role": "user", "content": prompt}],
                check={"type": "ifeval", "constraints": constraints},
                meta={"origin": tag, "constraint_type": r["constraint_type"]}, base=base)
    print("if_multi by origin", n)


# ---------------------------------------------------------------------- structured


def structured():
    """Structured-Outputs v2 split 1 (JSON/XML/YAML, synthetic documents) and v1
    (JSON). Splits 2 and 3 of v2 embed Wikipedia articles and are excluded."""
    f = out("structured")
    n = 0
    pf = pq.ParquetFile(f"{R}/nvidia__Nemotron-RL-Instruction-Following-Structured-Outputs-v2/"
                        "direct_generation/train-00000-of-00001.parquet")
    i = 0
    for b in pf.iter_batches(batch_size=1000, columns=["responses_create_params", "schema_str", "schema_type"]):
        for r in b.to_pylist():
            i += 1
            inp = r["responses_create_params"]["input"]
            if any(m["role"] not in ("user", "system") for m in inp):
                continue
            msgs = merge_same_role([{"role": m["role"], "content": m["content"]} for m in inp])
            h = hashlib.sha256((r["schema_str"] + json.dumps(msgs)).encode()).hexdigest()[:20]
            rec(f, key=f"so_v2:{h}", src="so_v2", kind="structured", messages=msgs,
                check={"type": "structured", "format": r["schema_type"].lower(), "schema": r["schema_str"]},
                meta={"row": i - 1})
            n += 1
    for i, line in enumerate(open(f"{R}/nvidia__Nemotron-RL-instruction_following-structured_outputs/"
                                  "structured_outputs_251027_nano_v3_sdg_json_train.jsonl")):
        r = json.loads(line)
        inp = r["responses_create_params"]["input"]
        msgs = merge_same_role([{"role": m["role"], "content": m["content"]} for m in inp])
        h = hashlib.sha256((r["schema_str"] + json.dumps(msgs)).encode()).hexdigest()[:20]
        rec(f, key=f"so_v1:{h}", src="so_v1", kind="structured", messages=msgs,
            check={"type": "structured", "format": r["schema_type"].lower(), "schema": r["schema_str"]},
            meta={"row": i})
        n += 1
    print("structured", n)


# -------------------------------------------------------------------------- safety


def safety():
    f = out("safety")
    n = 0
    for line in open(f"{R}/nvidia__Nemotron-SFT-Safety-v1/data/train.jsonl"):
        r = json.loads(line)
        if r["license"] != "cc-by-4.0":
            continue
        u = r["messages"][0]
        assert u["role"] == "user"
        rec(f, key=f"safety_v1:{r['uuid'][:24]}", src="safety_v1", kind="safety",
            messages=[{"role": "user", "content": u["content"]}])
        n += 1
    for r in json.load(open(f"{R}/nvidia__Aegis-AI-Content-Safety-Dataset-2.0/train.json")):
        if r["prompt"] == "REDACTED" or r["reconstruction_id_if_redacted"] is not None:
            continue
        rec(f, key=f"aegis2:{r['id']}", src="aegis2", kind="safety",
            messages=[{"role": "user", "content": r["prompt"]}],
            meta={"prompt_label": r["prompt_label"], "categories": r["violated_categories"]})
        n += 1
    print("safety", n)


# ------------------------------------------------------------------------ identity

OTHER_ASSISTANT = re.compile(r"\b(OLMo|Tülu|Tulu|AI2|Ai2|Allen Institute|Nemotron)\b", re.I)


def identity():
    import langid

    f = out("identity")
    seen = set()
    n = {"nemotron": 0, "olmo": 0}
    for line in open(f"{R}/nvidia__Nemotron-RL-Identity-Following-v1/train.jsonl"):
        r = json.loads(line)
        inp = r["responses_create_params"]["input"]
        if len(inp) != 1 or inp[0]["role"] != "user":
            continue
        text = inp[0]["content"].strip()
        if langid.classify(text)[0] != "en" or not text.isascii():
            continue
        if OTHER_ASSISTANT.search(text):
            continue
        k = text.lower()
        if k in seen:
            continue
        seen.add(k)
        rec(f, key=f"identity_nv:{hashlib.sha256(text.encode()).hexdigest()[:20]}", src="identity_nv",
            kind="identity", messages=[{"role": "user", "content": text}], check={"type": "identity"})
        n["nemotron"] += 1
    for line in open(f"{R}/allenai__olmo-2-hard-coded/hard_coded.jsonl"):
        r = json.loads(line)
        u = [m for m in r["messages"] if m["role"] == "user"]
        if len(u) != 1:
            continue
        text = u[0]["content"].strip()
        if OTHER_ASSISTANT.search(text):
            continue
        k = text.lower()
        if k in seen:
            continue
        seen.add(k)
        rec(f, key=f"identity_olmo:{r['id']}", src="identity_olmo", kind="identity",
            messages=[{"role": "user", "content": text}], check={"type": "identity"})
        n["olmo"] += 1
    print("identity", n)


# --------------------------------------------------------------------------- tools

_PY_TYPES = {
    "str": "string", "string": "string", "int": "integer", "integer": "integer", "float": "number",
    "number": "number", "bool": "boolean", "boolean": "boolean", "dict": "object", "list": "array",
    "tuple": "array", "set": "array", "any": None, "object": "object", "array": "array",
}


def _xlam_type(t: str):
    t = t.strip()
    base = t.split(",")[0].strip()
    m = re.match(r"^(List|list|Tuple|tuple|Set|set|Dict|dict)\[(.*)\]$", base)
    if m:
        outer = _PY_TYPES[m.group(1).lower()]
        if outer == "array":
            inner = _xlam_type(m.group(2).split(",")[0])
            return {"type": "array", "items": inner} if inner else {"type": "array"}
        return {"type": "object"}
    if base.lower() in ("union", "optional") or base.startswith(("Union", "Optional")):
        return {}
    kind = _PY_TYPES.get(base.lower())
    if kind is None and base.lower() not in ("any",):
        raise ValueError(base)
    return {"type": kind} if kind else {}


def xlam_tool(tool):
    """xLAM writes Python type names and marks optional parameters in the type
    string; rewrite as a JSON schema so every tool block speaks one dialect."""
    props, required = {}, []
    for pname, p in (tool.get("parameters") or {}).items():
        schema = _xlam_type(p.get("type", "str"))
        if p.get("description"):
            schema["description"] = p["description"]
        if "default" in p and p["default"] not in ("", None):
            schema["default"] = p["default"]
        props[pname] = schema
        if "optional" not in p.get("type", "").lower() and "default" not in p:
            required.append(pname)
    return {"name": tool["name"], "description": tool.get("description", ""),
            "parameters": {"type": "object", "properties": props, "required": required}}


def xlam():
    f = out("xlam")
    n = bad = 0
    for r in json.load(open(f"{R}/Salesforce__xlam-function-calling-60k/xlam_function_calling_60k.json")):
        try:
            tools = [xlam_tool(t) for t in json.loads(r["tools"])]
        except (ValueError, KeyError):
            bad += 1
            continue
        calls = json.loads(r["answers"])
        names = {t["name"] for t in tools}
        if not calls or any(c["name"] not in names for c in calls):
            bad += 1
            continue
        rec(f, key=f"xlam:{r['id']}", src="xlam", kind="tool_call",
            messages=[{"role": "user", "content": r["query"]}], tools=tools,
            check={"type": "tool_calls", "calls": [{"name": c["name"], "arguments": c["arguments"]} for c in calls]},
            meta={"id": r["id"]})
        n += 1
    print("xlam", n, "unconvertible", bad)


def _responses_to_chat(items):
    """Responses-API items -> chat messages. Past reasoning is dropped (the chat
    template drops it too); an assistant text and the calls that follow it
    become one assistant turn."""
    msgs = []
    for it in items:
        t = it.get("type")
        role = it.get("role")
        if t == "reasoning":
            continue
        if t is None and role in ("system", "user", "assistant"):
            content = it["content"] if isinstance(it["content"], str) else "".join(
                c.get("text", "") for c in it["content"])
            msgs.append({"role": role, "content": content})
        elif t == "message":
            content = "".join(c.get("text", "") for c in it["content"]) if isinstance(it["content"], list) else it["content"]
            msgs.append({"role": "assistant", "content": content})
        elif t == "function_call":
            args = json.loads(it["arguments"]) if isinstance(it["arguments"], str) else it["arguments"]
            call = {"id": it.get("call_id"), "name": it["name"], "arguments": args}
            if msgs and msgs[-1]["role"] == "assistant":
                msgs[-1].setdefault("tool_calls", []).append(call)
            else:
                msgs.append({"role": "assistant", "content": "", "tool_calls": [call]})
        elif t == "function_call_output":
            msgs.append({"role": "tool", "content": it["output"] if isinstance(it["output"], str)
                         else json.dumps(it["output"]), "tool_call_id": it.get("call_id")})
        else:
            raise ValueError(f"unknown item {t} {role}")
    if msgs and msgs[0]["role"] == "system" and not msgs[0]["content"].strip():
        msgs = msgs[1:]
    return msgs


def _tools(raw):
    tools = []
    for t in raw:
        params = t.get("parameters")
        if isinstance(params, str):
            params = json.loads(params)
        tools.append({"name": t["name"], "description": t.get("description", ""), "parameters": params or {}})
    return tools


def fc_pivot():
    f = out("fc_pivot")
    n = {"function_call": 0, "message": 0}
    for line in open(f"{R}/nvidia__Nemotron-RL-Agentic-Function-Calling-Pivot-v1/train.jsonl"):
        r = json.loads(line)
        msgs = _responses_to_chat(r["responses_create_params"]["input"])
        if not msgs or msgs[-1]["role"] not in ("user", "tool"):
            continue
        tools = _tools(r["responses_create_params"]["tools"])
        ea = r["expected_action"]
        if ea["type"] == "function_call":
            args = json.loads(ea["arguments"]) if isinstance(ea["arguments"], str) else ea["arguments"]
            check = {"type": "tool_calls", "calls": [{"name": ea["name"], "arguments": args}]}
        else:
            check = {"type": "no_tool_call"}
        info = r["info"]
        key = f"fc_pivot:{r['trajectory_id']}:{info['turn']}:{info['step']}"
        rec(f, key=key, src="fc_pivot", kind="tool_pivot", messages=msgs, tools=tools, check=check,
            meta={"trajectory": r["trajectory_id"], "expected": ea["type"]})
        n[ea["type"]] += 1
    print("fc_pivot", n)




# ------------------------------------------------------------------ toucan, conv pivot


def _toucan_msgs(raw):
    msgs = []
    for m in raw:
        role = m["role"]
        if role in ("user", "assistant", "system"):
            msgs.append({"role": role, "content": m["content"] or ""})
        elif role == "tool_call":
            c = ast.literal_eval(m["content"])
            args = c["arguments"]
            args = json.loads(args) if isinstance(args, str) and args.strip() else (args or {})
            call = {"name": c["name"], "arguments": args}
            if msgs and msgs[-1]["role"] == "assistant":
                msgs[-1].setdefault("tool_calls", []).append(call)
            else:
                msgs.append({"role": "assistant", "content": "", "tool_calls": [call]})
        elif role == "tool_response":
            msgs.append({"role": "tool", "content": m["content"] if isinstance(m["content"], str) else json.dumps(m["content"])})
        else:
            raise ValueError(role)
    return msgs


def toucan():
    """Toucan SFT subset as next-action pivots. single-turn: the first call after
    the question; multi-turn: the first call after the last user turn;
    irrelevant: no call at all (the shuffled servers cannot answer)."""
    import glob
    f = out("toucan")
    n = {}
    for p in sorted(glob.glob(f"{R}/Agent-Ark__Toucan-1.5M/SFT/*.parquet")):
        pf = pq.ParquetFile(p)
        for b in pf.iter_batches(batch_size=1000, columns=["uuid", "subset_name", "tools", "messages"]):
            for r in b.to_pylist():
                sub = r["subset_name"]
                try:
                    msgs = _toucan_msgs(json.loads(r["messages"]))
                    tools = [t["function"] if "function" in t else t for t in json.loads(r["tools"])]
                except (ValueError, SyntaxError, KeyError, TypeError):
                    n["unparsable"] = n.get("unparsable", 0) + 1
                    continue
                tools = [{"name": t["name"], "description": t.get("description", ""),
                          "parameters": t.get("parameters") or {}} for t in tools]
                msgs = [m for m in msgs if m["role"] != "system"]
                users = [i for i, m in enumerate(msgs) if m["role"] == "user"]
                if not users:
                    continue
                cut = users[-1] if sub == "multi-turn" else users[0]
                if sub == "multi-turn" and len(users) < 2:
                    continue
                nxt = msgs[cut + 1] if cut + 1 < len(msgs) else None
                if nxt is None or nxt["role"] != "assistant":
                    continue
                if sub == "irrelevant":
                    if nxt.get("tool_calls"):
                        continue
                    check = {"type": "no_tool_call"}
                else:
                    if not nxt.get("tool_calls"):
                        continue
                    check = {"type": "tool_calls", "calls": nxt["tool_calls"][:1], "first_of_many": len(nxt["tool_calls"]) > 1}
                ctx = msgs[:cut + 1]
                rec(f, key=f"toucan:{r['uuid']}", src="toucan", kind="tool_pivot" if sub == "multi-turn" else "tool_call",
                    messages=ctx, tools=tools, check=check, meta={"subset": sub})
                n[sub] = n.get(sub, 0) + 1
    print("toucan", n)


TAU2_DOMAIN = re.compile(r"\b(airline|airlines|flight|flights|aviation|retail|e-?commerce|telecom|telecommunications?|"
                         r"mobile (?:carrier|plan|phone)|wireless|cellular|phone plan|data plan|roaming)\b", re.I)


def conv_pivot():
    """Streams the 1.75 GB file once over HTTP, keeps a hashed 8% sample, drops
    every trajectory whose policy or tools are about τ²-bench's domains."""
    import requests
    from huggingface_hub import hf_hub_url
    tok = open(os.path.expanduser("~/.config/reliquary/hf-subnet.token")).read().strip()
    u = hf_hub_url("nvidia/Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1", "train.jsonl",
                   repo_type="dataset", revision="bfc7a4dd55ec1424cc3761ceacdd85a5743b4f4e")
    f = out("conv_pivot")
    n = {"rows": 0, "sampled": 0, "tau2_domain": 0, "kept_call": 0, "kept_message": 0, "bad": 0}
    with requests.get(u, headers={"Authorization": f"Bearer {tok}"}, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            n["rows"] += 1
            r = json.loads(line)
            tid = r["trajectory_id"]
            if int(hashlib.sha256(f"cp:{tid}".encode()).hexdigest()[:8], 16) % 100 >= 8:
                continue
            n["sampled"] += 1
            rcp = r["responses_create_params"]
            sysmsg = next((m["content"] for m in rcp["input"] if m.get("role") == "system"), "")
            title = re.search(r"#\s*([^\n]*?)\s*Policy", sysmsg)
            tools = _tools(rcp["tools"])
            probe = (title.group(1) if title else sysmsg[:600]) + " " + " ".join(
                t["name"] + " " + t["description"][:200] for t in tools)
            if TAU2_DOMAIN.search(probe):
                n["tau2_domain"] += 1
                continue
            try:
                msgs = _responses_to_chat(rcp["input"])
            except (ValueError, KeyError):
                n["bad"] += 1
                continue
            if not msgs or msgs[-1]["role"] not in ("user", "tool"):
                n["bad"] += 1
                continue
            ea = r["expected_action"]
            if ea["type"] == "function_call":
                args = json.loads(ea["arguments"]) if isinstance(ea["arguments"], str) else ea["arguments"]
                check = {"type": "tool_calls", "calls": [{"name": ea["name"], "arguments": args}]}
                n["kept_call"] += 1
            else:
                check = {"type": "no_tool_call"}
                n["kept_message"] += 1
            mi = r["meta_info"]
            rec(f, key=f"conv_pivot:{tid}:{mi['turn']}:{mi['step']}", src="conv_pivot", kind="tool_pivot",
                messages=msgs, tools=tools, check=check,
                meta={"trajectory": tid, "policy": title.group(1) if title else None,
                      "pass_rate": r.get("pass_rate"), "expected": ea["type"]})
    print("conv_pivot", n)

if __name__ == "__main__":
    for name in sys.argv[1:]:
        globals()[name]()
