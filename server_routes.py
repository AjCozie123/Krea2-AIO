"""HTTP route: read a LoRA's trigger/trained words from its safetensors metadata.

The node UI (js/krea_aio.js -> fetchTriggers) calls GET /kaio/lora_triggers?name=<lora>
when a LoRA is added, to auto-fill its trigger-word field. Best-effort — returns an empty
string if the file has no usable metadata. Registered on ComfyUI's PromptServer.
"""

import json
import logging
import os
import struct

log = logging.getLogger(__name__)


def _read_trigger_words(path):
    """Parse the safetensors header's __metadata__ and pull out trigger/trained words."""
    try:
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            if n <= 0 or n > 50_000_000:
                return ""
            header = json.loads(f.read(n).decode("utf-8", "ignore"))
        meta = header.get("__metadata__", {}) or {}

        # Direct trigger fields used by various trainers / model cards.
        for k in ("modelspec.trigger_phrase", "trigger_phrase", "ss_trigger_words",
                  "trigger_words", "activation text", "activation_text"):
            v = meta.get(k)
            if v and str(v).strip():
                return str(v).strip()

        # ai-toolkit / kohya tag frequency: {"dataset": {"tag": count}} -> most common tags.
        tf = meta.get("ss_tag_frequency")
        if tf:
            try:
                d = json.loads(tf) if isinstance(tf, str) else tf
                counts = {}
                for ds in (d.values() if isinstance(d, dict) else []):
                    if isinstance(ds, dict):
                        for tag, c in ds.items():
                            counts[tag] = counts.get(tag, 0) + int(c)
                top = sorted(counts, key=counts.get, reverse=True)[:5]
                top = [t.strip() for t in top if t and t.strip()]
                if top:
                    return ", ".join(top)
            except Exception:
                pass
    except Exception as e:
        log.debug("[KreaAIO] trigger read failed for %s: %s", path, e)
    return ""


def register_routes():
    try:
        import server  # ComfyUI's PromptServer module
        import folder_paths
        from aiohttp import web
    except Exception as e:
        log.warning("[KreaAIO] could not register lora-trigger route: %s", e)
        return

    inst = getattr(server.PromptServer, "instance", None)
    if inst is None:
        return

    @inst.routes.get("/kaio/prompt_dest")
    async def _prompt_dest(request):
        """Resolve the folder the Prompt Log node would write to, for the live preview.

        The panel must show the REAL path before anything runs, and only the server knows
        where ComfyUI's output directory is. Calls the same resolve_folder() the node uses,
        so the preview can never disagree with what actually happens.
        """
        q = request.rel_url.query
        try:
            from . import prompt_log
            idx = int(q.get("workflow") or 4)
            folder = prompt_log.resolve_folder(q.get("destination", ""),
                                               q.get("custom_folder", ""), idx)
            exists = os.path.isdir(folder)
            count = 0
            if exists:
                try:
                    count = sum(1 for f in os.listdir(folder)
                                if os.path.isfile(os.path.join(folder, f)))
                except OSError:
                    pass
            return web.json_response({"folder": folder, "exists": exists, "files": count})
        except Exception as e:
            log.debug("[KreaAIO] prompt_dest failed: %s", e)
            return web.json_response({"folder": "", "exists": False, "files": 0,
                                      "error": str(e)})

    @inst.routes.get("/kaio/lora_triggers")
    async def _lora_triggers(request):
        name = request.rel_url.query.get("name", "")
        triggers = ""
        try:
            path = folder_paths.get_full_path("loras", name) if name else None
            if path and os.path.exists(path):
                triggers = _read_trigger_words(path)
        except Exception:
            pass
        return web.json_response({"triggers": triggers})
