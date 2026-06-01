from __future__ import annotations

import argparse
import html
import json
import re
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs

from policy_recommendation_engine.analysis_modes import ANALYSIS_MODES, build_pipeline
from policy_recommendation_engine.database import (
    default_database_path,
    get_database_summary,
    list_analysis_runs,
    save_pipeline_result,
)
from policy_recommendation_engine.models import PipelineResult
from policy_recommendation_engine.network_graph import build_embedding_graph
from policy_recommendation_engine.upload import documents_from_pasted_text, documents_from_upload


DEFAULT_POLICY_PRIORITIES = {"water": 0.05, "healthcare": 0.2, "transport": 0.1}


class PolicyEngineHandler(BaseHTTPRequestHandler):
    server_version = "PolicyEngineWeb/0.1"

    def do_GET(self) -> None:
        self._send_html(render_page())

    def do_POST(self) -> None:
        try:
            documents, priorities, mode = self._parse_submission()
            if not documents:
                self._send_html(render_page(error="Add pasted text or upload a .txt, .md, .csv, or .pdf file.", mode=mode))
                return
            result = build_pipeline(mode).run(documents, policy_priorities=priorities)
            saved_run_id = save_pipeline_result(result, analysis_mode=mode, policy_priorities=priorities)
            self._send_html(
                render_page(
                    result=result,
                    priorities=priorities,
                    mode=mode,
                    saved_run_id=saved_run_id,
                )
            )
        except Exception as exc:  # noqa: BLE001 - surface friendly local UI errors.
            self._send_html(render_page(error=str(exc)), status=HTTPStatus.BAD_REQUEST)

    def log_message(self, format: str, *args: object) -> None:
        return

    def _parse_submission(self) -> tuple[tuple[object, ...], dict[str, float], str]:
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        content_type = self.headers.get("Content-Type", "")

        if content_type.startswith("multipart/form-data"):
            fields, files = parse_multipart(body, content_type)
            pasted_text = fields.get("pasted_text", "")
            documents = list(documents_from_pasted_text(pasted_text))
            for filename, content in files:
                documents.extend(documents_from_upload(filename, content))
            return (
                tuple(documents),
                parse_policy_priorities(fields.get("policy_priorities", "")),
                parse_analysis_mode(fields.get("analysis_mode", "")),
            )

        fields = parse_qs(body.decode("utf-8"))
        documents = documents_from_pasted_text(fields.get("pasted_text", [""])[0])
        return (
            documents,
            parse_policy_priorities(fields.get("policy_priorities", [""])[0]),
            parse_analysis_mode(fields.get("analysis_mode", [""])[0]),
        )

    def _send_html(self, content: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        encoded = content.encode("utf-8")
        try:
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            return


def parse_policy_priorities(raw: str) -> dict[str, float]:
    if not raw.strip():
        return DEFAULT_POLICY_PRIORITIES
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("Policy priorities must be valid JSON, for example {\"water\": 0.05}.") from exc

    priorities: dict[str, float] = {}
    for key, value in parsed.items():
        priorities[str(key).lower()] = float(value)
    return priorities


def parse_analysis_mode(raw: str) -> str:
    mode = raw.strip() or "lightweight"
    if mode not in ANALYSIS_MODES:
        raise ValueError(f"Unknown analysis mode: {mode}")
    return mode


def parse_multipart(body: bytes, content_type: str) -> tuple[dict[str, str], list[tuple[str, bytes]]]:
    boundary_match = re.search(r"boundary=(?P<boundary>[^;]+)", content_type)
    if not boundary_match:
        raise ValueError("Upload request is missing a multipart boundary.")

    boundary = ("--" + boundary_match.group("boundary").strip('"')).encode("utf-8")
    fields: dict[str, str] = {}
    files: list[tuple[str, bytes]] = []

    for part in body.split(boundary):
        part = part.strip()
        if not part or part == b"--":
            continue
        header_blob, _, value = part.partition(b"\r\n\r\n")
        if not value:
            continue
        value = value.removesuffix(b"\r\n").removesuffix(b"--")
        headers = header_blob.decode("utf-8", errors="replace")
        name = _header_attribute(headers, "name")
        filename = _header_attribute(headers, "filename")
        if not name:
            continue
        if filename:
            if value:
                files.append((filename, value))
        else:
            fields[name] = value.decode("utf-8", errors="replace")

    return fields, files


def _header_attribute(headers: str, attribute: str) -> str | None:
    match = re.search(rf'{attribute}="(?P<value>[^"]*)"', headers)
    return match.group("value") if match else None


def render_page(
    *,
    result: PipelineResult | None = None,
    priorities: dict[str, float] | None = None,
    mode: str = "lightweight",
    saved_run_id: int | None = None,
    error: str | None = None,
) -> str:
    priorities_json = html.escape(json.dumps(priorities or DEFAULT_POLICY_PRIORITIES, indent=2))
    history = list_analysis_runs(limit=12)
    archive_summary = get_database_summary()
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Policy Intelligence Engine</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --ink: #172026;
      --muted: #607080;
      --line: #d9e0e6;
      --accent: #0b6b63;
      --accent-strong: #084f49;
      --warn: #b54708;
      --bad: #b42318;
      font-family: Arial, Helvetica, sans-serif;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: var(--bg); color: var(--ink); }}
    header {{ background: #ffffff; border-bottom: 1px solid var(--line); }}
    .wrap {{ width: min(1180px, calc(100% - 32px)); margin: 0 auto; }}
    .top {{ display: flex; align-items: center; justify-content: space-between; min-height: 72px; gap: 16px; }}
    h1 {{ font-size: 24px; margin: 0; letter-spacing: 0; }}
    main {{ padding: 24px 0 40px; }}
    form {{ display: grid; grid-template-columns: minmax(0, 1.25fr) minmax(280px, 0.75fr); gap: 16px; align-items: start; }}
    section, .panel {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 16px; }}
    label {{ display: block; font-weight: 700; margin-bottom: 8px; }}
    textarea {{ width: 100%; min-height: 210px; resize: vertical; border: 1px solid var(--line); border-radius: 6px; padding: 12px; font: inherit; }}
    input[type="file"] {{ width: 100%; border: 1px dashed var(--line); padding: 14px; border-radius: 6px; background: #fbfcfd; }}
    .upload-meter {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-top: 12px; }}
    .upload-meter div {{ border: 1px solid var(--line); border-radius: 8px; padding: 10px; background: #fbfcfd; }}
    .upload-meter strong {{ display: block; font-size: 20px; }}
    .mode-list {{ display: grid; gap: 8px; }}
    .mode-option {{ border: 1px solid var(--line); border-radius: 8px; padding: 10px; display: grid; grid-template-columns: 20px 1fr; gap: 8px; align-items: start; }}
    .mode-option input {{ margin-top: 3px; }}
    .mode-option span {{ display: block; font-weight: 700; }}
    button {{ border: 0; border-radius: 6px; background: var(--accent); color: white; padding: 11px 14px; font-weight: 700; cursor: pointer; }}
    button:hover {{ background: var(--accent-strong); }}
    .stack {{ display: grid; gap: 16px; }}
    .hint {{ color: var(--muted); font-size: 14px; margin: 8px 0 0; line-height: 1.45; }}
    .success {{ border-color: #a7d7c5; background: #eefaf5; color: #075e45; margin-bottom: 16px; }}
    .error {{ border-color: #f5c2bd; background: #fff4f2; color: var(--bad); margin-bottom: 16px; }}
    .results {{ display: grid; gap: 16px; margin-top: 18px; }}
    .grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    th, td {{ text-align: left; border-bottom: 1px solid var(--line); padding: 10px 8px; vertical-align: top; }}
    th {{ color: var(--muted); font-weight: 700; }}
    .metric {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }}
    .metric div {{ border: 1px solid var(--line); border-radius: 8px; padding: 12px; }}
    .metric strong {{ display: block; font-size: 24px; }}
    ul {{ margin: 0; padding-left: 20px; }}
    code {{ background: #edf2f4; border-radius: 4px; padding: 2px 4px; }}
    .history-table td:last-child {{ max-width: 360px; }}
    .graph-wrap {{ display: grid; gap: 12px; }}
    .embedding-graph {{ width: 100%; height: auto; min-height: 320px; border: 1px solid var(--line); border-radius: 8px; background: #fbfcfd; }}
    .graph-node text {{ fill: #ffffff; font-size: 12px; font-weight: 700; pointer-events: none; }}
    .graph-legend {{ display: flex; flex-wrap: wrap; gap: 8px 14px; color: var(--muted); font-size: 14px; }}
    .graph-legend span {{ display: inline-flex; align-items: center; gap: 6px; }}
    .graph-legend i {{ width: 12px; height: 12px; border-radius: 50%; border: 1px solid #17323a; display: inline-block; }}
    @media (max-width: 850px) {{
      form, .grid, .metric, .upload-meter {{ grid-template-columns: 1fr; }}
      .top {{ align-items: flex-start; flex-direction: column; padding: 16px 0; }}
    }}
  </style>
</head>
<body>
  <header>
    <div class="wrap top">
      <h1>Policy Intelligence Engine</h1>
      <span class="hint">Upload public feedback, extract themes, emotions, gaps, and trends.</span>
    </div>
  </header>
  <main class="wrap">
    {render_error(error)}
    {render_saved_message(saved_run_id)}
    <form method="post" enctype="multipart/form-data">
      <section>
        <label for="pasted_text">Paste public comments or transcripts</label>
        <textarea id="pasted_text" name="pasted_text" placeholder="Paste one or more comments. Separate documents with a blank line."></textarea>
        <p class="hint">Supported uploads: <code>.txt</code>, <code>.md</code>, <code>.pdf</code>, and CSV files with a <code>text</code> column.</p>
        <div class="upload-meter" aria-live="polite">
          <div><span class="hint">Pasted Docs</span><strong id="pasted_count">0</strong></div>
          <div><span class="hint">Selected Files</span><strong id="file_count">0</strong></div>
          <div><span class="hint">Estimated Input</span><strong id="total_count">0</strong></div>
        </div>
      </section>
      <div class="stack">
        <section>
          <label for="upload">Upload data file</label>
          <input id="upload" name="upload" type="file" accept=".txt,.md,.csv,.pdf" multiple>
        </section>
        <section>
          <label>Analysis mode</label>
          <div class="mode-list">
            {render_mode_options(mode)}
          </div>
          <p class="hint">spaCy and BERT modes use real NLP libraries and may take longer on first run.</p>
        </section>
        <section>
          <label for="policy_priorities">Policy priorities JSON</label>
          <textarea id="policy_priorities" name="policy_priorities" style="min-height: 132px;">{priorities_json}</textarea>
          <p class="hint">Values are shares from 0 to 1. Example: water at 0.05 means 5% policy attention.</p>
        </section>
        <button type="submit">Analyze</button>
      </div>
    </form>
    {render_results(result)}
    {render_history(history, archive_summary)}
  </main>
  <script>
    const pastedText = document.getElementById("pasted_text");
    const uploadInput = document.getElementById("upload");
    const pastedCount = document.getElementById("pasted_count");
    const fileCount = document.getElementById("file_count");
    const totalCount = document.getElementById("total_count");

    function countPastedDocuments(value) {{
      const trimmed = value.trim();
      if (!trimmed) {{
        return 0;
      }}
      return trimmed.split(/\\n\\s*\\n/).filter(Boolean).length;
    }}

    function updateInputCounts() {{
      const pasted = countPastedDocuments(pastedText.value);
      const files = uploadInput.files.length;
      pastedCount.textContent = pasted;
      fileCount.textContent = files;
      totalCount.textContent = pasted + files;
    }}

    pastedText.addEventListener("input", updateInputCounts);
    uploadInput.addEventListener("change", updateInputCounts);
    updateInputCounts();
  </script>
</body>
</html>"""


def render_error(error: str | None) -> str:
    if not error:
        return ""
    return f'<section class="error">{html.escape(error)}</section>'


def render_saved_message(saved_run_id: int | None) -> str:
    if saved_run_id is None:
        return ""
    database_path = html.escape(str(default_database_path()))
    return f'<section class="success">Saved analysis run #{saved_run_id} to <code>{database_path}</code>.</section>'


def render_results(result: PipelineResult | None) -> str:
    if result is None:
        return ""

    return f"""
    <div class="results">
      <section class="metric">
        <div><span class="hint">Documents</span><strong>{len(result.documents)}</strong></div>
        <div><span class="hint">Themes</span><strong>{len(result.themes)}</strong></div>
        <div><span class="hint">Policy Gaps</span><strong>{len(result.policy_gaps)}</strong></div>
      </section>
      <div class="grid">
        <section>
          <h2>Themes</h2>
          {render_themes_table(result)}
        </section>
        <section>
          <h2>Emotion Map</h2>
          {render_emotions_table(result)}
        </section>
      </div>
      <section>
        <h2>Policy Gaps</h2>
        {render_gaps_table(result)}
      </section>
      <section>
        <h2>Embedding Network</h2>
        {render_embedding_graph(result)}
      </section>
      <section>
        <h2>Named Entities</h2>
        {render_entities_table(result)}
      </section>
      <section>
        <h2>Insights</h2>
        <ul>{''.join(f'<li>{html.escape(item)}</li>' for item in result.insights)}</ul>
      </section>
    </div>"""


def render_embedding_graph(result: PipelineResult) -> str:
    graph = build_embedding_graph(result)
    edge_count = len(graph.edges)
    node_count = len(graph.nodes)
    return (
        f'<p class="hint">{node_count} documents connected by {edge_count} embedding-similarity links. '
        "Thicker lines mean stronger semantic similarity.</p>"
        f"{graph.svg}"
    )


def render_history(history: list[dict[str, object]], archive_summary: dict[str, int]) -> str:
    if not history:
        return """
        <section class="results">
          <h2>Process History</h2>
          <p class="hint">No archived analysis runs yet.</p>
        </section>"""

    rows = "".join(render_history_row(run) for run in history)
    total_runs = archive_summary.get("analysis_runs", 0)
    total_documents = archive_summary.get("documents", 0)
    total_themes = archive_summary.get("themes", 0)
    return f"""
    <section class="results">
      <h2>Process History</h2>
      <section class="metric">
        <div><span class="hint">Archived Runs</span><strong>{total_runs}</strong></div>
        <div><span class="hint">Archived Docs</span><strong>{total_documents}</strong></div>
        <div><span class="hint">Archived Themes</span><strong>{total_themes}</strong></div>
      </section>
      <table class="history-table">
        <thead>
          <tr>
            <th>Run</th>
            <th>Created</th>
            <th>Mode</th>
            <th>Docs</th>
            <th>Themes</th>
            <th>Top Themes</th>
            <th>Insight Preview</th>
          </tr>
        </thead>
        <tbody>{rows}</tbody>
      </table>
    </section>"""


def render_history_row(run: dict[str, object]) -> str:
    themes = run.get("themes", [])
    insights = run.get("insights", [])
    theme_text = ", ".join(str(theme) for theme in themes) if themes else "None"
    insight_text = " ".join(str(insight) for insight in insights) if insights else "None"
    return (
        "<tr>"
        f"<td>#{run['id']}</td>"
        f"<td>{html.escape(str(run['created_at']))}</td>"
        f"<td>{html.escape(str(run['analysis_mode']))}</td>"
        f"<td>{run['document_count']}</td>"
        f"<td>{run['theme_count']}</td>"
        f"<td>{html.escape(theme_text)}</td>"
        f"<td>{html.escape(insight_text)}</td>"
        "</tr>"
    )


def render_mode_options(selected: str) -> str:
    options: list[str] = []
    for mode in ANALYSIS_MODES.values():
        checked = " checked" if mode.key == selected else ""
        options.append(
            '<label class="mode-option">'
            f'<input type="radio" name="analysis_mode" value="{html.escape(mode.key)}"{checked}>'
            "<div>"
            f"<span>{html.escape(mode.label)}</span>"
            f'<p class="hint">{html.escape(mode.description)}</p>'
            "</div>"
            "</label>"
        )
    return "".join(options)


def render_themes_table(result: PipelineResult) -> str:
    rows = "".join(
        "<tr>"
        f"<td>{html.escape(theme.name)}</td>"
        f"<td>{html.escape(', '.join(theme.keywords))}</td>"
        f"<td>{len(theme.document_indexes)}</td>"
        "</tr>"
        for theme in result.themes
    )
    return f"<table><thead><tr><th>Theme</th><th>Keywords</th><th>Docs</th></tr></thead><tbody>{rows}</tbody></table>"


def render_emotions_table(result: PipelineResult) -> str:
    rows = "".join(
        "<tr>"
        f"<td>{html.escape(theme)}</td>"
        f"<td>{html.escape(signal.dominant_emotion)}</td>"
        f"<td>{html.escape(signal.intensity)}</td>"
        "</tr>"
        for theme, signal in result.emotions_by_theme.items()
    )
    return f"<table><thead><tr><th>Theme</th><th>Emotion</th><th>Intensity</th></tr></thead><tbody>{rows}</tbody></table>"


def render_gaps_table(result: PipelineResult) -> str:
    rows = "".join(
        "<tr>"
        f"<td>{html.escape(gap.theme)}</td>"
        f"<td>{gap.public_share:.0%}</td>"
        f"<td>{gap.policy_share:.0%}</td>"
        f"<td>{gap.gap_score:.0%}</td>"
        f"<td>{html.escape(gap.severity)}</td>"
        "</tr>"
        for gap in result.policy_gaps
    )
    return (
        "<table><thead><tr><th>Theme</th><th>Public Share</th><th>Policy Share</th>"
        f"<th>Gap</th><th>Severity</th></tr></thead><tbody>{rows}</tbody></table>"
    )


def render_entities_table(result: PipelineResult) -> str:
    rows: list[str] = []
    for index, document in enumerate(result.documents, start=1):
        for text, label in document.named_entities:
            rows.append(
                "<tr>"
                f"<td>{index}</td>"
                f"<td>{html.escape(text)}</td>"
                f"<td>{html.escape(label)}</td>"
                "</tr>"
            )
    if not rows:
        return '<p class="hint">No named entities found. Choose a spaCy mode to enable entity extraction.</p>'
    return "<table><thead><tr><th>Doc</th><th>Entity</th><th>Label</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local policy intelligence web UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), PolicyEngineHandler)
    print(f"Policy Intelligence Engine running at http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.")
    server.serve_forever()


if __name__ == "__main__":
    main()
