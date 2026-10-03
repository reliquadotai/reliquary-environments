"""One wire fix for the benchmarks' runs, applied in every interpreter `run.py` starts.

verifiers' chat mediation rewrites a request's ``tools: null`` (what the null
harness sends for a task without tools) into ``tools: []``, and vLLM refuses an
empty tools array with a 400 on every request. A request without tools must say
so by leaving the field out. Loaded through ``PYTHONPATH``, so the env-server
and interception processes verifiers spawns carry it too; it touches nothing
when verifiers is not installed.
"""


def _drop_empty_tools() -> None:
    try:
        from verifiers.v1.dialects.chat import ChatDialect
    except Exception:
        return
    original = ChatDialect.mediate_external_capabilities
    if getattr(original, "_heldout_wirefix", False):
        return

    def mediate(self, body, policy):
        mediated, capabilities = original(self, body, policy)
        if isinstance(mediated, dict) and mediated.get("tools") == []:
            mediated.pop("tools")
            # A tool choice with no tools is refused the same way.
            if mediated.get("tool_choice") in ("auto", "none", None):
                mediated.pop("tool_choice", None)
        return mediated, capabilities

    mediate._heldout_wirefix = True
    ChatDialect.mediate_external_capabilities = mediate


_drop_empty_tools()
