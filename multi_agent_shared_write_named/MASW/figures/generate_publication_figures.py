"""Generate publication-style SVG figures for MASW.

The script keeps coordinates on a small grid and uses shared drawing helpers so
the figures are visually consistent. It intentionally generates flat 2D vector
graphics rather than bitmap images.
"""

from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape


OUT_DIR = Path(__file__).resolve().parent
W, H = 1600, 900

COLORS = {
    "ink": "#1F2933",
    "muted": "#667085",
    "blue": "#D0E0F0",
    "blue2": "#BFD4E9",
    "peach": "#FAD7C5",
    "mint": "#DDEFE6",
    "gray": "#EFF2F5",
    "memory": "#F5F1E8",
    "line": "#222222",
    "red": "#B85C5C",
    "green": "#5D8F72",
}


def header(title: str, subtitle: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        "<defs>",
        '<marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">',
        '<path d="M0,0 L8,3 L0,6 Z" fill="#222222"/>',
        "</marker>",
        "<style>",
        ".title{font-family:Arial,Helvetica,sans-serif;font-size:32px;font-weight:700;fill:#1F2933}",
        ".subtitle{font-family:Arial,Helvetica,sans-serif;font-size:17px;fill:#667085}",
        ".panel{font-family:Arial,Helvetica,sans-serif;font-size:22px;font-weight:700;fill:#1F2933}",
        ".label{font-family:Arial,Helvetica,sans-serif;font-size:17px;font-weight:700;fill:#1F2933}",
        ".small{font-family:Arial,Helvetica,sans-serif;font-size:13.5px;fill:#344054}",
        ".tiny{font-family:Arial,Helvetica,sans-serif;font-size:12px;fill:#475467}",
        ".axis{stroke:#D0D5DD;stroke-width:1}",
        ".line{stroke:#222222;stroke-width:1.45;fill:none;marker-end:url(#arrow)}",
        ".softline{stroke:#475467;stroke-width:1.2;fill:none;stroke-dasharray:7 5;marker-end:url(#arrow)}",
        ".dash{stroke:#667085;stroke-width:1.2;fill:none;stroke-dasharray:7 5}",
        ".op{fill:#FFFFFF;stroke:#222222;stroke-width:1.3}",
        "</style>",
        "</defs>",
        '<rect width="1600" height="900" fill="#FFFFFF"/>',
        f'<text x="64" y="58" class="title">{escape(title)}</text>',
        f'<text x="64" y="88" class="subtitle">{escape(subtitle)}</text>',
    ]


def close(parts: list[str]) -> str:
    return "\n".join(parts + ["</svg>\n"])


def rect(x: int, y: int, w: int, h: int, fill: str, text: str = "", sub: str = "", cls: str = "label") -> str:
    lines = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{fill}" stroke="#2F3A45" stroke-width="1.25"/>']
    if text:
        lines.append(f'<text x="{x + 18}" y="{y + 38}" class="{cls}">{escape(text)}</text>')
    if sub:
        for i, line in enumerate(sub.split("|")):
            lines.append(f'<text x="{x + 18}" y="{y + 68 + 22 * i}" class="tiny">{escape(line)}</text>')
    return "\n".join(lines)


def pill(x: int, y: int, w: int, text: str, fill: str) -> str:
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="34" rx="17" fill="{fill}" stroke="#2F3A45" stroke-width="1.1"/>'
        f'<text x="{x + 16}" y="{y + 23}" class="tiny">{escape(text)}</text>'
    )


def arrow(x1: int, y1: int, x2: int, y2: int, dashed: bool = False) -> str:
    cls = "softline" if dashed else "line"
    return f'<path d="M{x1} {y1} H{x2}" class="{cls}"/>'


def manhattan(points: list[tuple[int, int]], dashed: bool = False) -> str:
    cls = "softline" if dashed else "line"
    d = f"M{points[0][0]} {points[0][1]} " + " ".join(f"L{x} {y}" for x, y in points[1:])
    return f'<path d="{d}" class="{cls}"/>'


def tensor_strip(x: int, y: int, label: str, fill: str) -> str:
    parts = []
    for i in range(5):
        parts.append(f'<rect x="{x + i * 18}" y="{y + i * 7}" width="140" height="72" rx="8" fill="{fill}" stroke="#2F3A45" stroke-width="1.0"/>')
    parts.append(f'<text x="{x + 30}" y="{y + 116}" class="label">{escape(label)}</text>')
    return "\n".join(parts)


def op_circle(x: int, y: int, label: str) -> str:
    return f'<circle cx="{x}" cy="{y}" r="18" class="op"/><text x="{x - 6}" y="{y + 6}" class="label">{escape(label)}</text>'


def legend(items: list[tuple[str, str]]) -> str:
    parts = [f'<rect x="64" y="812" width="1472" height="52" rx="12" fill="{COLORS["gray"]}" stroke="#D0D5DD" stroke-width="1"/>']
    x = 96
    parts.append('<text x="96" y="845" class="label">Legend</text>')
    x += 130
    for name, color in items:
        parts.append(f'<rect x="{x}" y="827" width="36" height="20" rx="6" fill="{color}" stroke="#2F3A45" stroke-width="1"/>')
        parts.append(f'<text x="{x + 48}" y="843" class="tiny">{escape(name)}</text>')
        x += 230
    return "\n".join(parts)


def fig1() -> str:
    p = header(
        "MASW Defense Architecture",
        "Provenance-aware shared-memory pipeline with mediated execution",
    )
    p.append('<text x="64" y="138" class="panel">A. Secure write path</text>')
    p.append(tensor_strip(72, 225, "External Data", COLORS["mint"]))
    xs = [300, 480, 660, 840, 1050]
    labels = [
        ("D1 Input Label", "trust=UNTRUSTED|taint=true", COLORS["blue"]),
        ("D2 Candidate", "fact extraction|no direct write", COLORS["blue"]),
        ("D3 Risk Filter", "hybrid detector|early screening", COLORS["peach"]),
        ("D4 Provenance Gate", "verify before promote|single write gate", COLORS["peach"]),
        ("Verified Memory", "shared state|evidence only", COLORS["memory"]),
    ]
    for x, (t, s, c) in zip(xs, labels):
        p.append(rect(x, 245, 150 if x < 840 else 175, 108, c, t, s))
    p.extend([
        arrow(220, 290, 300, 290),
        arrow(450, 290, 480, 290),
        arrow(630, 290, 660, 290),
        arrow(810, 290, 840, 290),
        arrow(1015, 290, 1050, 290),
    ])
    p.append('<text x="64" y="482" class="panel">B. Secure read and action path</text>')
    p.append(rect(1050, 560, 175, 108, COLORS["memory"], "Verified Memory", "retrieval source|scoped access"))
    p.append(rect(820, 560, 170, 108, COLORS["blue"], "D5 Retrieval Audit", "trust-aware ranking|taint filtering"))
    p.append(rect(590, 560, 170, 108, COLORS["blue"], "Agent B Planner", "proposal only|not authorization"))
    p.append(rect(360, 560, 170, 108, COLORS["peach"], "D6 Mediator", "tool policy|context risk"))
    p.append(rect(120, 560, 170, 108, COLORS["mint"], "Tool API", "email / db / repo|dry-run executor"))
    p.extend([
        arrow(1050, 614, 990, 614),
        arrow(820, 614, 760, 614),
        arrow(590, 614, 530, 614),
        arrow(360, 614, 290, 614),
    ])
    p.append(rect(1260, 245, 235, 110, COLORS["gray"], "Core Invariants", "No automatic trust promotion|Memory is evidence|Proposal is not authorization"))
    p.append(manhattan([(1172, 245), (1172, 180), (1260, 180), (1260, 245)], dashed=True))
    p.append(legend([
        ("processing node", COLORS["blue"]),
        ("security gate", COLORS["peach"]),
        ("memory state", COLORS["memory"]),
        ("external interface", COLORS["mint"]),
    ]))
    return close(p)


def fig2() -> str:
    p = header(
        "MASW-Poison-Exec Attack Chain",
        "Before: flat trust enables delayed tool misuse; After: MASW breaks the chain",
    )
    p.append('<text x="64" y="148" class="panel">A. Vulnerable flat-trust system</text>')
    top = [("Q_inject", COLORS["peach"]), ("Agent A", COLORS["blue"]), ("Poisoned Memory", COLORS["memory"]), ("Q_target", COLORS["mint"]), ("Agent B", COLORS["blue"]), ("Dangerous Tool", COLORS["peach"])]
    xs = [80, 315, 550, 785, 1020, 1255]
    for x, (t, c) in zip(xs, top):
        p.append(rect(x, 210, 170, 102, c, t, ""))
    for x in xs[:-1]:
        p.append(arrow(x + 170, 261, x + 235, 261))
    p.append(pill(560, 338, 250, "wrong rule: T_mem(x) = T_writer(A)", COLORS["gray"]))
    p.append(pill(1255, 338, 235, "observed execution: 8 / 8", COLORS["peach"]))

    p.append('<text x="64" y="455" class="panel">B. MASW protected system</text>')
    bot = [("Q_inject", COLORS["peach"]), ("D1-D2", COLORS["blue"]), ("D3-D4", COLORS["peach"]), ("Quarantine", COLORS["gray"]), ("D5", COLORS["blue"]), ("D6 Deny", COLORS["peach"])]
    for x, (t, c) in zip(xs, bot):
        p.append(rect(x, 520, 170, 102, c, t, ""))
    for i, x in enumerate(xs[:-1]):
        p.append(arrow(x + 170, 571, x + 235, 571, dashed=i >= 3))
    p.append('<path d="M1260 500 L1465 640 M1465 500 L1260 640" stroke="#B85C5C" stroke-width="4" stroke-linecap="round"/>')
    p.append(pill(552, 648, 260, "poisoned memory writes: 0 / 8", COLORS["mint"]))
    p.append(pill(1255, 648, 250, "dangerous executions: 0 / 8", COLORS["mint"]))

    p.append(rect(164, 730, 1272, 72, COLORS["gray"], "Attack-chain factorization", "P_success = P_write  x  P_retrieve  x  P_execute"))
    p.append(op_circle(650, 775, "x"))
    p.append(op_circle(840, 775, "x"))
    return close(p)


def fig3() -> str:
    p = header("D3 Hybrid Risk Filter", "Rule precision plus semantic recall for early-stage screening")
    p.append(tensor_strip(80, 360, "CandidateFact", COLORS["mint"]))
    p.append(rect(360, 220, 250, 135, COLORS["blue"], "RuleBasedDetector", "override patterns|tool commands|policy-memory terms"))
    p.append(rect(360, 505, 250, 135, COLORS["peach"], "RubricDetector", "role semantics|future-agent control|external sink"))
    p.append(op_circle(725, 430, "max"))
    p.append(rect(835, 365, 220, 130, COLORS["gray"], "Hybrid Score", "R = max(rule, rubric)|tau_write = 0.45"))
    p.append(op_circle(1145, 430, ">"))
    p.append(rect(1240, 260, 220, 105, COLORS["peach"], "Block", "quarantine|audit reason"))
    p.append(rect(1240, 500, 220, 105, COLORS["mint"], "Pass", "forward to D4|retain provenance"))
    p.extend([
        manhattan([(250, 430), (310, 430), (310, 288), (360, 288)]),
        manhattan([(250, 430), (310, 430), (310, 573), (360, 573)]),
        manhattan([(610, 288), (670, 288), (670, 430), (707, 430)]),
        manhattan([(610, 573), (670, 573), (670, 430), (707, 430)]),
        arrow(743, 430, 835, 430),
        arrow(1055, 430, 1127, 430),
        manhattan([(1163, 430), (1200, 430), (1200, 312), (1240, 312)]),
        manhattan([(1163, 430), (1200, 430), (1200, 552), (1240, 552)]),
    ])
    p.append(rect(930, 660, 430, 82, COLORS["gray"], "Selection Result", "D3_RULE recall = 0.8889 | D3_HYBRID recall = 1.0000"))
    p.append(legend([
        ("rule features", COLORS["blue"]),
        ("semantic rubric", COLORS["peach"]),
        ("candidate tensor", COLORS["mint"]),
    ]))
    return close(p)


def fig4() -> str:
    p = header("D4 Provenance Gate", "Trust promotion requires evidence, source reputation, conflict checks, and low risk")
    p.append(tensor_strip(80, 360, "Candidate", COLORS["mint"]))
    checks = [
        ("Evidence", "evidence_ok(c)", 190),
        ("Source", "rep(src) >= rho_min", 320),
        ("Conflict", "conflict_free(c)", 450),
        ("Risk", "risk(c) <= tau_verify", 580),
    ]
    for t, s, y in checks:
        p.append(rect(360, y, 230, 85, COLORS["blue"] if t != "Risk" else COLORS["peach"], t, s))
        p.append(manhattan([(250, 430), (315, 430), (315, y + 42), (360, y + 42)]))
        p.append(manhattan([(590, y + 42), (670, y + 42), (670, 430), (710, 430)]))
    p.append(op_circle(735, 430, "AND"))
    p.append(rect(850, 350, 290, 155, COLORS["gray"], "verify(c)", "evidence_ok AND source_ok|AND conflict_free|AND risk <= tau_verify"))
    p.append(rect(1260, 230, 220, 105, COLORS["memory"], "Verified Memory", "trust = VERIFIED|taint = false"))
    p.append(rect(1260, 540, 220, 105, COLORS["peach"], "Quarantine", "reason + provenance|no retrieval"))
    p.append(arrow(753, 430, 850, 430))
    p.append(manhattan([(1140, 405), (1205, 405), (1205, 282), (1260, 282)]))
    p.append(manhattan([(1140, 470), (1205, 470), (1205, 592), (1260, 592)]))
    p.append(rect(80, 650, 390, 82, COLORS["gray"], "Belief Score", "0.50 trust + 0.30 evidence + 0.20 source reputation"))
    p.append(rect(850, 650, 440, 82, COLORS["gray"], "Node-Aware Placement", "D4 uses RuleBasedDetector; D4 hybrid FPR = 0.0364"))
    return close(p)


def fig5() -> str:
    p = header("Evaluation Protocol", "RQ1-RQ6: node tests, ablation, baselines, latency, and end-to-end validation")
    p.append(rect(70, 190, 250, 160, COLORS["mint"], "Dataset", "125 minimum rows|+20 hard-set rows|8 E2E scenarios"))
    blocks = [
        (430, 160, "RQ1 Unit Tests", "43 tests passed", COLORS["blue"]),
        (430, 330, "RQ2 Regression", "attack writes = 0", COLORS["blue"]),
        (430, 500, "RQ3 Ablation", "ALL F1 = 1.0000", COLORS["blue"]),
        (760, 160, "RQ4 Selection", "D3 hybrid + D4 rule", COLORS["peach"]),
        (760, 330, "RQ5 Baselines", "3 tool baselines", COLORS["peach"]),
        (760, 500, "RQ6 E2E", "MASW exec = 0 / 8", COLORS["peach"]),
    ]
    for x, y, t, s, c in blocks:
        p.append(rect(x, y, 240, 104, c, t, s))
        p.append(manhattan([(320, 270), (370, 270), (370, y + 52), (430, y + 52)] if x == 430 else [(670, y + 52), (760, y + 52)]))
    p.append(rect(1120, 180, 360, 170, COLORS["gray"], "Metrics", "Precision, Recall, F1, FPR|P50 / P95 / P99 ms|TP, FP, FN, TN"))
    p.append(rect(1120, 450, 360, 170, COLORS["gray"], "Key Results", "MASW F1 = 1.0000|tool baseline recall <= 0.4333|vulnerable E2E exec = 8 / 8"))
    p.append(manhattan([(1000, 212), (1070, 212), (1070, 265), (1120, 265)]))
    p.append(manhattan([(1000, 552), (1070, 552), (1070, 535), (1120, 535)]))
    p.append(rect(110, 720, 1360, 82, COLORS["gray"], "Result Artifacts", "ablation_metrics.csv | tech_selection_metrics.csv | baseline_comparison_metrics.csv | e2e_attack_validation_report.md"))
    return close(p)


FIGURES = {
    "fig1_masw_overall_architecture.svg": fig1,
    "fig2_attack_chain_vs_defense.svg": fig2,
    "fig3_d3_hybrid_detector.svg": fig3,
    "fig4_d4_provenance_gate.svg": fig4,
    "fig5_evaluation_protocol.svg": fig5,
}


def main() -> None:
    for name, build in FIGURES.items():
        (OUT_DIR / name).write_text(build(), encoding="utf-8")


if __name__ == "__main__":
    main()
