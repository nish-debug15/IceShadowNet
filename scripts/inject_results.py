"""
scripts/inject_results.py
=========================
Copies the generated training summaries into reports/report_skeleton.md
between the "Auto-populated" marker and the next section heading.

Usage:
    python -m scripts.inject_results
"""


def demote_headings(markdown, levels=3):
    """Push '#' headings down so a standalone summary nests under a report subsection."""
    out = []
    for line in markdown.splitlines():
        if line.startswith("#"):
            line = "#" * levels + line
        out.append(line)
    return "\n".join(out)


def inject(report_path, tag_start, tag_end, content_file):
    with open(report_path, 'r', encoding='utf-8') as f:
        text = f.read()
    # Summaries are written as UTF-8 by train_all.py / train_multimodal.py.
    # Older runs on Windows wrote cp1252, so fall back to that if needed.
    try:
        with open(content_file, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        with open(content_file, 'r', encoding='cp1252', errors='replace') as f:
            content = f.read()
    content = demote_headings(content.strip())

    start_idx = text.find(tag_start)
    end_idx = text.find(tag_end, start_idx)

    if start_idx == -1 or end_idx == -1:
        print(f"WARNING: markers for {content_file} not found in {report_path}; skipped.")
        return
    new_text = text[:start_idx + len(tag_start)] + '\n\n' + content + '\n\n' + text[end_idx:]
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(new_text)
    print(f"Injected {content_file} into {report_path}")


if __name__ == "__main__":
    inject('reports/report_skeleton.md', '<!-- Auto-populated from reports/training_summary.md -->', '### 6.2', 'reports/training_summary.md')
    inject('reports/report_skeleton.md', '<!-- Auto-populated from reports/training_summary_multimodal.md -->', '### 6.3', 'reports/training_summary_multimodal.md')
