"""Krea2 Prompt Log — save and organise the prompts the AIO node actually sampled with.

Why this exists
---------------
The AIO node builds its final prompt internally (LLM enhancement, then trigger words),
so the text that reached the sampler is not the text you typed and, until now, was not
recoverable after the run. This node takes that finished prompt and files it on disk.

Design notes, taken from the nodes that already do this well:

* WAS Suite's "Save Text File" is the de-facto standard: a path, a filename prefix, a
  delimiter and a zero-padded counter, creating the directory if it is missing. That
  counter is what stops a second run silently overwriting the first, so it is kept.
* WAS also supports [time(...)] tokens in the path. Useful, but easy to get wrong by
  hand, so the same result is offered as a plain "add a date subfolder" choice instead.
* ComfyUI-Prompt-Stash keeps its library under ComfyUI/user/ and organises prompts into
  named lists reached from a dropdown. The "organise into categories" idea is right; the
  categories here are the four Krea 2 workflows, chosen automatically.
* Prompt-Stash's pass-through is worth copying: this node returns the prompt unchanged,
  so it can sit inline rather than as a dead end.

The default destination is  ComfyUI/output/prompts/<workflow>/  with one folder per
workflow, so prompts sit beside the images they produced without any configuration.
"""

import csv
import json
import logging
import os
import re
import time

import folder_paths

from comfy_api.latest import io

log = logging.getLogger(__name__)


# The four workflow folders. Numbered so they sort in pipeline order in Explorer.
WORKFLOW_FOLDERS = {
    1: "1_Classic",
    2: "2_Identity",
    3: "3_Inpaint_Outpaint",
    4: "4_TextToImage",
}

WORKFLOW_CHOICES = [
    "Auto - from the AIO node",
    "1 - Classic edit",
    "2 - Identity / Ostris edit",
    "3 - Inpaint / Outpaint",
    "4 - Text to image",
]

DESTINATIONS = [
    "Auto - output/prompts/<workflow>",
    "Auto + a date subfolder",
    "Custom folder (set below)",
]

FILE_MODES = [
    "One file per prompt",
    "Append to a daily log",
    "Append to one master log",
]

FORMATS = [
    "Text (.txt)",
    "Markdown (.md)",
    "JSON Lines (.jsonl)",
    "Spreadsheet (.csv)",
]

EXTENSIONS = {
    "Text (.txt)": ".txt",
    "Markdown (.md)": ".md",
    "JSON Lines (.jsonl)": ".jsonl",
    "Spreadsheet (.csv)": ".csv",
}

# Windows reserved device names — a file called con.txt or nul.txt cannot be created.
_RESERVED = {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)),
             *(f"lpt{i}" for i in range(1, 10))}


def safe_name(name, fallback="prompt"):
    """Make a string safe to use as a filename component on Windows and POSIX."""
    name = str(name or "").strip()
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name)     # illegal on Windows
    name = name.strip(" .")                                 # trailing dot/space is illegal
    if not name or name.lower() in _RESERVED:
        return fallback
    return name[:80]


def resolve_folder(destination, custom_folder, idx):
    """Work out the directory a prompt will be written to.

    Kept as a plain function with no side effects so the node UI can show the user the
    exact resolved path BEFORE they run anything — the whole point of the panel.
    """
    if str(destination).startswith("Custom") and str(custom_folder or "").strip():
        base = os.path.abspath(os.path.expanduser(str(custom_folder).strip()))
    else:
        base = os.path.join(folder_paths.get_output_directory(), "prompts",
                            WORKFLOW_FOLDERS.get(idx, "4_TextToImage"))
    if str(destination).startswith("Auto + a date"):
        base = os.path.join(base, time.strftime("%Y-%m-%d"))
    return base


def next_numbered_path(folder, prefix, ext, padding=4):
    """prefix_0001.txt, prefix_0002.txt … never overwriting an existing file.

    Scans the folder once and continues from the highest number already there, which is
    what stops a restarted ComfyUI from writing over yesterday's prompts.
    """
    pat = re.compile(re.escape(prefix) + r"_(\d+)" + re.escape(ext) + r"$", re.IGNORECASE)
    highest = 0
    try:
        for f in os.listdir(folder):
            m = pat.match(f)
            if m:
                highest = max(highest, int(m.group(1)))
    except OSError:
        pass
    return os.path.join(folder, f"{prefix}_{highest + 1:0{padding}d}{ext}")


def _rows(meta, prompt):
    """The fields written for a prompt, in a fixed order shared by every format."""
    return [
        ("timestamp", time.strftime("%Y-%m-%d %H:%M:%S")),
        ("workflow", meta.get("workflow_name", "")),
        ("prompt", prompt),
        ("typed", meta.get("prompt_typed", "")),
        ("enhanced", meta.get("prompt_enhanced", "")),
        ("enhancer_used", "yes" if meta.get("enhancer_on") else "no"),
        ("enhancer_changed", "yes" if meta.get("enhancer_changed") else "no"),
        ("trigger_words", meta.get("trigger_words", "")),
        ("negative", meta.get("negative_prompt", "")),
        ("seed", meta.get("seed", "")),
        ("steps", meta.get("steps", "")),
        ("cfg", meta.get("cfg", "")),
        ("sampler", meta.get("sampler", "")),
        ("scheduler", meta.get("scheduler", "")),
        ("model", meta.get("unet_name", "")),
        ("encoder", meta.get("clip_name", "")),
        ("loras", meta.get("loras", "")),
        ("save_path", meta.get("save_path", "")),
    ]


def render(fmt, prompt, meta, include_metadata):
    """Format one entry. Returns (text, is_structured)."""
    if fmt.startswith("JSON"):
        d = dict(_rows(meta, prompt)) if include_metadata else {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "prompt": prompt}
        return json.dumps(d, ensure_ascii=False), True
    if fmt.startswith("Spreadsheet"):
        return None, True          # CSV is written through the csv module, below
    if not include_metadata:
        return prompt, False
    if fmt.startswith("Markdown"):
        out = [f"## {time.strftime('%Y-%m-%d %H:%M:%S')} — {meta.get('workflow_name','')}", "",
               prompt, ""]
        for k, v in _rows(meta, prompt):
            if k in ("timestamp", "workflow", "prompt") or v in ("", None):
                continue
            out.append(f"- **{k}**: {v}")
        return "\n".join(out) + "\n", False
    # plain text
    out = []
    for k, v in _rows(meta, prompt):
        if v in ("", None):
            continue
        out.append(f"{k}: {v}" if k != "prompt" else f"prompt:\n{v}")
    return "\n".join(out) + "\n", False


class Krea2PromptLog(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Krea2PromptLog",
            display_name="Krea2 Prompt Log (AJ)",
            category="KREA2",
            description=(
                "Save the prompt the Krea2 AIO node actually sampled with. Files it into "
                "output/prompts/<workflow> automatically, or anywhere you choose. Passes the "
                "prompt straight through so it can sit inline."
            ),
            is_output_node=True,
            inputs=[
                io.String.Input("prompt", optional=True, force_input=True,
                                tooltip="Wire the AIO node's `prompt` output here."),
                io.String.Input("prompt_json", optional=True, force_input=True,
                                tooltip="Wire the AIO node's `prompt_json` output here. Carries "
                                        "the workflow, seed, sampler and LoRA stack so the log "
                                        "can file itself and record the settings."),

                io.Boolean.Input("enabled", default=True,
                                 tooltip="Off = write nothing. The prompt still passes through."),
                io.Combo.Input("destination", options=DESTINATIONS, default=DESTINATIONS[0],
                               tooltip="Auto files each workflow into its own folder under "
                                       "ComfyUI/output/prompts."),
                io.String.Input("custom_folder", default="",
                                tooltip="Only used by 'Custom folder'. An absolute path, or one "
                                        "relative to the ComfyUI folder. Created if missing."),
                io.Combo.Input("workflow", options=WORKFLOW_CHOICES, default=WORKFLOW_CHOICES[0],
                               tooltip="Which of the four folders to use. Auto reads it from the "
                                       "AIO node, so you never have to keep this in sync."),
                io.Combo.Input("file_mode", options=FILE_MODES, default=FILE_MODES[0],
                               tooltip="One file per prompt is easiest to browse; the append "
                                       "modes keep a single running log you can search."),
                io.Combo.Input("file_format", options=FORMATS, default=FORMATS[0]),
                io.String.Input("filename_prefix", default="prompt",
                                tooltip="Illegal filename characters are replaced automatically."),
                io.Boolean.Input("include_metadata", default=True,
                                 tooltip="Record seed, steps, cfg, sampler, model and LoRAs "
                                         "alongside the prompt. Off = just the prompt text."),
                io.Boolean.Input("skip_duplicates", default=True,
                                 tooltip="Do not write again if this is the same prompt as the "
                                         "last one saved here — stops a re-queue filling the "
                                         "folder with identical files."),
            ],
            outputs=[
                io.String.Output(display_name="prompt"),
                io.String.Output(display_name="saved_to"),
            ],
        )

    # Last prompt written per destination file, for skip_duplicates.
    _last = {}

    @classmethod
    def execute(cls, enabled=True, destination=DESTINATIONS[0], custom_folder="",
                workflow=WORKFLOW_CHOICES[0], file_mode=FILE_MODES[0],
                file_format=FORMATS[0], filename_prefix="prompt",
                include_metadata=True, skip_duplicates=True,
                prompt=None, prompt_json=None) -> io.NodeOutput:

        meta = {}
        if prompt_json:
            try:
                meta = json.loads(prompt_json) or {}
            except Exception as e:
                log.warning("[Krea2PromptLog] prompt_json is not valid JSON (%s); logging the "
                            "prompt without its settings.", e)

        text = (prompt if prompt is not None else meta.get("prompt_final", "")) or ""
        text = text.strip()

        # Workflow index: the explicit pick wins, otherwise whatever the AIO reported.
        if workflow.startswith("Auto"):
            idx = int(meta.get("pipeline") or 4)
        else:
            idx = int(workflow[0])
        meta.setdefault("workflow_name", WORKFLOW_FOLDERS.get(idx, "4_TextToImage"))

        if not enabled:
            return io.NodeOutput(text, "", ui={"krea2_log": [{"saved": "", "skipped": "disabled"}]})
        if not text:
            log.info("[Krea2PromptLog] nothing to save — the prompt is empty.")
            return io.NodeOutput(text, "", ui={"krea2_log": [{"saved": "", "skipped": "empty"}]})

        folder = resolve_folder(destination, custom_folder, idx)
        ext = EXTENSIONS.get(file_format, ".txt")
        prefix = safe_name(filename_prefix)

        try:
            os.makedirs(folder, exist_ok=True)
        except OSError as e:
            log.error("[Krea2PromptLog] cannot create %s (%s) — nothing was saved.", folder, e)
            return io.NodeOutput(text, "", ui={"krea2_log": [{"saved": "", "error": str(e)}]})

        if file_mode.startswith("One file"):
            path = next_numbered_path(folder, prefix, ext)
            append = False
        elif file_mode.startswith("Append to a daily"):
            path = os.path.join(folder, f"{prefix}_{time.strftime('%Y-%m-%d')}{ext}")
            append = True
        else:
            path = os.path.join(folder, f"{prefix}_log{ext}")
            append = True

        if skip_duplicates and cls._last.get(path) == text:
            log.info("[Krea2PromptLog] identical to the last prompt saved here — skipped.")
            return io.NodeOutput(text, path,
                                 ui={"krea2_log": [{"saved": path, "skipped": "duplicate"}]})

        try:
            if file_format.startswith("Spreadsheet"):
                rows = _rows(meta, text) if include_metadata else [
                    ("timestamp", time.strftime("%Y-%m-%d %H:%M:%S")), ("prompt", text)]
                new = not os.path.exists(path) or not append
                # newline="" is required or the csv module writes blank lines on Windows.
                with open(path, "a" if append else "w", encoding="utf-8-sig", newline="") as f:
                    w = csv.writer(f)
                    if new:
                        w.writerow([k for k, _ in rows])
                    w.writerow([v for _, v in rows])
            else:
                body, _ = render(file_format, text, meta, include_metadata)
                # Only separate entries when appending to a file that already has content —
                # otherwise a fresh log starts with a stray rule or blank line.
                has_content = append and os.path.exists(path) and os.path.getsize(path) > 0
                separator = "" if file_format.startswith("JSON") else "\n" + "-" * 70 + "\n\n"
                with open(path, "a" if append else "w", encoding="utf-8") as f:
                    if has_content:
                        f.write(separator)
                    f.write(body if body.endswith("\n") else body + "\n")
        except OSError as e:
            log.error("[Krea2PromptLog] could not write %s (%s).", path, e)
            return io.NodeOutput(text, "", ui={"krea2_log": [{"saved": "", "error": str(e)}]})

        cls._last[path] = text
        log.info("[Krea2PromptLog] saved to %s", path)
        return io.NodeOutput(text, path, ui={"krea2_log": [{
            "saved": path,
            "folder": folder,
            "workflow": WORKFLOW_FOLDERS.get(idx, ""),
            "mode": file_mode,
            "format": file_format,
        }]})
