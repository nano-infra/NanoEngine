"""NanoFold CLI client — predict + visualize + LLM analysis with tool calling.

Usage:
    python -m NanoFold.client \\
        --sequence MKTAYIAKQRQISFVK \\
        --prompt "Focus on hydrophobicity and potential membrane interaction." \\
        --url http://127.0.0.1:3001 \\
        --model /models/models--Qwen--Qwen3.5-397B-A17B-FP8
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import sys
import time
from io import BytesIO
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Section 1 — FoldClient
# ═══════════════════════════════════════════════════════════════════════════════


class FoldClient:
    """HTTP client wrapping NanoRoute /v1/structure/* fold endpoints."""

    def __init__(
        self, base_url: str, timeout: float = 600.0, poll_interval: float = 3.0
    ):
        self._base = base_url.rstrip("/")
        self._timeout = timeout
        self._poll_interval = poll_interval

    # ── Public API ────────────────────────────────────────────────────────────

    def predict(
        self,
        sequence: str,
        name: str = "query",
        seeds: list[int] | None = None,
        n_sample: int = 1,
        n_step: int = 200,
    ) -> dict:
        """Submit predict job and poll until done. Returns job result dict."""
        body = {
            "name": name,
            "sequences": [{"proteinChain": {"sequence": sequence, "count": 1}}],
            "seeds": seeds or [101],
            "n_sample": n_sample,
            "n_step": n_step,
            "use_msa": False,
            "use_template": False,
        }
        resp = httpx.post(
            f"{self._base}/v1/structure/predict",
            json=body,
            timeout=self._timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        if not data.get("ok"):
            raise RuntimeError(f"predict submit failed: {data}")
        job_id = data["job_id"]
        logger.info("Submitted predict job %s", job_id)
        return self._poll_job(job_id)

    def embed(self, sequence: str, name: str = "query") -> str:
        """Submit embed job and poll until done. Returns embed_id."""
        body = {
            "name": name,
            "sequences": [{"proteinChain": {"sequence": sequence, "count": 1}}],
            "use_msa": False,
            "use_template": False,
        }
        resp = httpx.post(
            f"{self._base}/v1/structure/embed",
            json=body,
            timeout=self._timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        if not data.get("ok"):
            raise RuntimeError(f"embed submit failed: {data}")
        embed_id = data["embed_id"]
        logger.info("Submitted embed %s", embed_id)
        self._poll_embed(embed_id)
        return embed_id

    def sample(
        self,
        embed_id: str,
        seeds: list[int] | None = None,
        n_sample: int = 1,
        n_step: int = 200,
    ) -> dict:
        """Submit sample job for an existing embed and poll until done."""
        body = {
            "embed_id": embed_id,
            "seeds": seeds or [101],
            "n_sample": n_sample,
            "n_step": n_step,
        }
        resp = httpx.post(
            f"{self._base}/v1/structure/sample",
            json=body,
            timeout=self._timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        if not data.get("ok"):
            raise RuntimeError(f"sample submit failed: {data}")
        job_id = data["job_id"]
        logger.info("Submitted sample job %s", job_id)
        return self._poll_job(job_id)

    # ── Polling ───────────────────────────────────────────────────────────────

    def _poll_job(self, job_id: str, poll_interval: float | None = None) -> dict:
        interval = poll_interval if poll_interval is not None else self._poll_interval
        url = f"{self._base}/v1/structure/jobs/{job_id}"
        while True:
            resp = httpx.get(url, timeout=30.0)
            resp.raise_for_status()
            data = resp.json()
            status = data.get("status", "")
            logger.debug("Job %s status=%s", job_id, status)
            if status == "done":
                return data
            if status == "error":
                raise RuntimeError(f"Job {job_id} failed: {data.get('error')}")
            time.sleep(interval)

    def _poll_embed(self, embed_id: str, poll_interval: float | None = None) -> dict:
        interval = poll_interval if poll_interval is not None else self._poll_interval
        url = f"{self._base}/v1/structure/embeds/{embed_id}"
        while True:
            resp = httpx.get(url, timeout=30.0)
            resp.raise_for_status()
            data = resp.json()
            status = data.get("status", "")
            logger.debug("Embed %s status=%s", embed_id, status)
            if status == "done":
                return data
            if status == "error":
                raise RuntimeError(f"Embed {embed_id} failed: {data.get('error')}")
            time.sleep(interval)


# ═══════════════════════════════════════════════════════════════════════════════
# Section 2 — StructureRenderer
# ═══════════════════════════════════════════════════════════════════════════════


class StructureRenderer:
    """Render Cα backbone from mmCIF content to a PNG image (matplotlib)."""

    @staticmethod
    def render(cif_b64: str) -> bytes | None:
        """Decode base64 CIF, parse Cα coords, draw 3-D backbone. Returns PNG bytes or None."""
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        except ImportError:
            logger.warning("matplotlib unavailable; skipping structure rendering")
            return None

        cif_text = base64.b64decode(cif_b64).decode("utf-8", errors="replace")
        coords = StructureRenderer._parse_ca_coords(cif_text)
        if not coords:
            logger.warning("No Cα atoms found in CIF; skipping render")
            return None

        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        zs = [c[2] for c in coords]
        n = len(coords)
        colors = [i / max(n - 1, 1) for i in range(n)]

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(xs, ys, zs, linewidth=0.8, color="gray", alpha=0.5)
        sc = ax.scatter(xs, ys, zs, c=colors, cmap="rainbow", s=15, zorder=5)
        plt.colorbar(sc, ax=ax, label="Residue index (N→C)", shrink=0.5)
        ax.set_xlabel("X (Å)")
        ax.set_ylabel("Y (Å)")
        ax.set_zlabel("Z (Å)")
        ax.set_title(f"Cα backbone ({n} residues)")
        fig.tight_layout()

        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=120)
        plt.close(fig)
        buf.seek(0)
        return buf.read()

    @staticmethod
    def _parse_ca_coords(cif_text: str) -> list[tuple[float, float, float]]:
        """Extract Cα x/y/z from the _atom_site loop in an mmCIF file."""
        coords: list[tuple[float, float, float]] = []
        lines = cif_text.splitlines()
        i = 0
        n = len(lines)

        while i < n:
            line = lines[i].strip()

            if line != "loop_":
                i += 1
                continue

            i += 1
            # Collect column definitions (lines starting with "_")
            col_names: list[str] = []
            while i < n:
                col_line = lines[i].strip()
                if col_line.startswith("_"):
                    col_names.append(col_line.split()[0])
                    i += 1
                else:
                    break

            # Only care about _atom_site loops
            if not any(c.startswith("_atom_site.") for c in col_names):
                continue

            # Map column names to indices
            col_x = col_y = col_z = col_atom = -1
            for j, name in enumerate(col_names):
                if name == "_atom_site.Cartn_x":
                    col_x = j
                elif name == "_atom_site.Cartn_y":
                    col_y = j
                elif name == "_atom_site.Cartn_z":
                    col_z = j
                elif name == "_atom_site.label_atom_id":
                    col_atom = j

            if col_x < 0 or col_y < 0 or col_z < 0:
                continue

            min_cols = max(col_x, col_y, col_z, col_atom if col_atom >= 0 else 0) + 1

            # Parse data rows until the loop ends
            while i < n:
                row = lines[i].strip()
                if (
                    not row
                    or row.startswith("_")
                    or row.startswith("#")
                    or row == "loop_"
                    or row.startswith("data_")
                ):
                    break
                i += 1
                tokens = row.split()
                if len(tokens) < min_cols:
                    continue
                if col_atom >= 0 and tokens[col_atom] != "CA":
                    continue
                try:
                    coords.append(
                        (
                            float(tokens[col_x]),
                            float(tokens[col_y]),
                            float(tokens[col_z]),
                        )
                    )
                except ValueError:
                    pass

        return coords


# ═══════════════════════════════════════════════════════════════════════════════
# Section 3 — NanoFoldAssistant
# ═══════════════════════════════════════════════════════════════════════════════

_TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "predict_structure",
            "description": "Predict the 3D structure of a protein sequence using AlphaFold3",
            "parameters": {
                "type": "object",
                "properties": {
                    "sequence": {
                        "type": "string",
                        "description": "Single-letter amino acid sequence",
                    },
                    "seeds": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "default": [101],
                    },
                    "n_step": {
                        "type": "integer",
                        "default": 200,
                    },
                },
                "required": ["sequence"],
            },
        },
    }
]


class NanoFoldAssistant:
    """Orchestrates: fold → render → LLM streaming chat with tool calling."""

    def __init__(
        self,
        base_url: str,
        model: str,
        fold_client: FoldClient,
        enable_image: bool = True,
        enable_tools: bool = True,
    ):
        self._base = base_url.rstrip("/")
        self._model = model
        self._fold = fold_client
        self._enable_image = enable_image
        self._enable_tools = enable_tools

    # ── Main entry ────────────────────────────────────────────────────────────

    def run(
        self,
        sequence: str,
        prompt: str,
        name: str = "query",
        seeds: list[int] | None = None,
        n_sample: int = 1,
        n_step: int = 200,
        save_cif: str | None = None,
        extra_image_url: str | None = None,
    ) -> None:
        """Predict → render → stream LLM analysis to stdout."""
        # ── Step 1: Fold ──────────────────────────────────────────────────────
        print(
            f"[NanoFold] Predicting structure for {len(sequence)}-residue sequence...",
            file=sys.stderr,
        )
        result = self._fold.predict(
            sequence, name=name, seeds=seeds or [101], n_sample=n_sample, n_step=n_step
        )

        structures = result.get("structures") or []
        confidence_list = result.get("confidence") or []

        if not structures:
            print("[NanoFold] No structures returned.", file=sys.stderr)
            return

        best_struct = structures[0]
        best_conf = confidence_list[0] if confidence_list else {}
        cif_b64 = best_struct.get("content", "")

        # Save CIF
        if save_cif and cif_b64:
            Path(save_cif).write_bytes(base64.b64decode(cif_b64))
            print(f"[NanoFold] CIF saved to {save_cif}", file=sys.stderr)

        # ── Step 2: Render ────────────────────────────────────────────────────
        image_b64: str | None = None
        if self._enable_image and cif_b64:
            png_bytes = StructureRenderer.render(cif_b64)
            if png_bytes:
                image_b64 = base64.b64encode(png_bytes).decode()
                print(
                    f"[NanoFold] Rendered Cα backbone ({len(png_bytes) // 1024} KB)",
                    file=sys.stderr,
                )

        # ── Step 3: Build initial messages ────────────────────────────────────
        plddt = best_conf.get("plddt", "N/A")
        ptm = best_conf.get("ptm", "N/A")
        iptm = best_conf.get("iptm", "N/A")
        ranking = best_conf.get("ranking_score", "N/A")
        has_clash = best_conf.get("has_clash", False)

        structure_context = (
            f"Sequence: {sequence}\n"
            f"Length: {len(sequence)} residues\n"
            f"pLDDT: {plddt}\n"
            f"pTM: {ptm}\n"
            f"ipTM: {iptm}\n"
            f"Ranking score: {ranking}\n"
            f"Has clash: {has_clash}\n"
            f"Job ID: {result.get('job_id', 'N/A')}"
        )

        system_msg: dict = {
            "role": "system",
            "content": (
                "You are an expert structural biologist with deep knowledge of protein folding, "
                "structure-function relationships, and computational structural biology. "
                "You are analyzing a structure prediction result from an AlphaFold3-based model.\n\n"
                f"Structure prediction summary:\n{structure_context}"
            ),
        }

        user_parts: list[dict] = [{"type": "text", "text": prompt}]
        if self._enable_image and image_b64:
            user_parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                }
            )
        if extra_image_url:
            user_parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": extra_image_url},
                }
            )

        if len(user_parts) == 1:
            user_msg: dict = {"role": "user", "content": prompt}
        else:
            user_msg = {"role": "user", "content": user_parts}

        messages: list[dict] = [system_msg, user_msg]
        tools = _TOOLS if self._enable_tools else None

        # ── Step 4: Chat loop ─────────────────────────────────────────────────
        print("[NanoFold] Analyzing with LLM...\n", file=sys.stderr)
        self._chat_loop(messages, tools)

    # ── Chat loop ─────────────────────────────────────────────────────────────

    def _chat_loop(self, messages: list[dict], tools: list[dict] | None) -> None:
        while True:
            content, tool_calls, finish_reason = self._chat(messages, tools)
            if finish_reason != "tool_calls" or not tool_calls:
                break
            # Append assistant turn
            asst_msg: dict[str, Any] = {"role": "assistant", "tool_calls": tool_calls}
            if content:
                asst_msg["content"] = content
            messages.append(asst_msg)
            # Execute tools and append results
            for tc in tool_calls:
                result = self._execute_tool(tc)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": json.dumps(result),
                    }
                )
            # Continue loop

    def _chat(
        self,
        messages: list[dict],
        tools: list[dict] | None,
    ) -> tuple[str, list[dict], str | None]:
        """Stream one chat completion. Returns (content, tool_calls, finish_reason)."""
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "stream": True,
        }
        if tools:
            payload["tools"] = tools

        accumulated_content = ""
        # index → {id, type, function: {name, arguments}}
        tc_accum: dict[int, dict] = {}
        finish_reason: str | None = None

        with httpx.Client(timeout=None) as client:
            with client.stream(
                "POST", f"{self._base}/v1/chat/completions", json=payload
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    data_str = line[len("data: ") :]
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    choices = chunk.get("choices", [])
                    if not choices:
                        continue
                    choice = choices[0]
                    delta = choice.get("delta", {})
                    fr = choice.get("finish_reason")
                    if fr:
                        finish_reason = fr

                    # Content delta → stream to stdout
                    if delta.get("content"):
                        text = delta["content"]
                        accumulated_content += text
                        print(text, end="", flush=True)

                    # Tool call deltas → accumulate silently
                    for tc_delta in delta.get("tool_calls", []):
                        idx = tc_delta.get("index", 0)
                        if idx not in tc_accum:
                            tc_accum[idx] = {
                                "id": "",
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            }
                        if tc_delta.get("id"):
                            tc_accum[idx]["id"] += tc_delta["id"]
                        fn = tc_delta.get("function", {})
                        if fn.get("name"):
                            tc_accum[idx]["function"]["name"] += fn["name"]
                        if fn.get("arguments"):
                            tc_accum[idx]["function"]["arguments"] += fn["arguments"]

        if accumulated_content:
            print()  # trailing newline

        tc_list = [tc_accum[i] for i in sorted(tc_accum)]
        return accumulated_content, tc_list, finish_reason

    def _execute_tool(self, tc: dict) -> dict:
        fn_name = tc.get("function", {}).get("name", "")
        try:
            args = json.loads(tc.get("function", {}).get("arguments", "{}"))
        except json.JSONDecodeError:
            return {"error": "invalid tool arguments JSON"}

        if fn_name == "predict_structure":
            seq = args.get("sequence", "")
            seeds = args.get("seeds", [101])
            n_step = args.get("n_step", 200)
            print(
                f"\n[Tool] predict_structure(sequence={seq[:20]}..., seeds={seeds}, n_step={n_step})",
                file=sys.stderr,
            )
            try:
                result = self._fold.predict(seq, seeds=seeds, n_step=n_step)
                conf_list = result.get("confidence") or []
                conf = conf_list[0] if conf_list else {}
                return {
                    "plddt": conf.get("plddt"),
                    "ptm": conf.get("ptm"),
                    "iptm": conf.get("iptm"),
                    "ranking_score": conf.get("ranking_score"),
                    "has_clash": conf.get("has_clash"),
                    "job_id": result.get("job_id"),
                }
            except Exception as exc:
                return {"error": str(exc)}

        return {"error": f"unknown tool: {fn_name!r}"}


# ═══════════════════════════════════════════════════════════════════════════════
# Section 4 — CLI
# ═══════════════════════════════════════════════════════════════════════════════


def _image_to_data_url(path: str) -> str:
    import mimetypes

    mime = mimetypes.guess_type(path)[0] or "image/png"
    data = base64.b64encode(Path(path).read_bytes()).decode()
    return f"data:{mime};base64,{data}"


def _extract_sequence_from_image(base_url: str, model: str, image_data_url: str) -> str:
    """Ask the LLM to extract a protein sequence from an image."""
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a bioinformatics assistant. "
                    "Extract the protein amino acid sequence from this image. "
                    "Return ONLY the single-letter amino acid sequence, nothing else."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                    {
                        "type": "text",
                        "text": "Extract the amino acid sequence from this image.",
                    },
                ],
            },
        ],
        "stream": False,
    }
    resp = httpx.post(url, json=payload, timeout=60.0)
    resp.raise_for_status()
    data = resp.json()
    raw = data["choices"][0]["message"]["content"]
    return raw.strip().replace("\n", "").replace(" ", "")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NanoFold CLI: predict protein structure + LLM analysis"
    )
    parser.add_argument(
        "--sequence",
        default=None,
        help="Amino acid sequence in single-letter code",
    )
    parser.add_argument(
        "--image",
        default=None,
        metavar="PATH",
        help=(
            "Path to an image file (PNG/JPG/etc.). "
            "If --sequence is absent, the LLM extracts the sequence from the image. "
            "The image is also included as context in the final analysis."
        ),
    )
    parser.add_argument("--name", default="query", help="Name for the prediction job")
    parser.add_argument(
        "--prompt",
        default="Analyze this protein structure prediction.",
        help="Analysis prompt for the LLM",
    )
    parser.add_argument(
        "--url", default="http://127.0.0.1:3001", help="NanoRoute base URL"
    )
    parser.add_argument("--model", required=True, help="LLM model path or name")
    parser.add_argument("--seeds", type=int, nargs="+", default=[101])
    parser.add_argument("--n_step", type=int, default=200)
    parser.add_argument("--n_sample", type=int, default=1)
    parser.add_argument("--poll_interval", type=float, default=3.0)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument(
        "--save_cif", default=None, metavar="PATH", help="Save CIF to file"
    )
    parser.add_argument(
        "--no_image",
        action="store_true",
        help="Disable Cα backbone image rendering (does not affect --image input)",
    )
    parser.add_argument("--no_tools", action="store_true", help="Disable tool calling")
    parser.add_argument("--log_level", default="warning")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    if not args.sequence and not args.image:
        parser.error("Provide at least one of --sequence or --image.")

    fold_client = FoldClient(
        base_url=args.url,
        timeout=args.timeout,
        poll_interval=args.poll_interval,
    )

    # Build image data URL once (used for both sequence extraction and analysis context)
    image_data_url: str | None = None
    if args.image:
        image_data_url = _image_to_data_url(args.image)

    sequence = args.sequence

    # If no sequence provided, extract it from the image via LLM vision
    if not sequence:
        print(
            f"[NanoFold] Extracting sequence from {args.image} via LLM...",
            file=sys.stderr,
        )
        sequence = _extract_sequence_from_image(args.url, args.model, image_data_url)
        print(f"[NanoFold] Extracted sequence: {sequence}", file=sys.stderr)

    assistant = NanoFoldAssistant(
        base_url=args.url,
        model=args.model,
        fold_client=fold_client,
        enable_image=not args.no_image,
        enable_tools=not args.no_tools,
    )

    assistant.run(
        sequence=sequence,
        prompt=args.prompt,
        name=args.name,
        seeds=args.seeds,
        n_sample=args.n_sample,
        n_step=args.n_step,
        save_cif=args.save_cif,
        # Pass image as extra visual context in the analysis message when --image was given
        extra_image_url=image_data_url,
    )


if __name__ == "__main__":
    main()
