import os

def inject(report_path, tag_start, tag_end, content_file):
    with open(report_path, 'r', encoding='utf-8') as f:
        text = f.read()
    with open(content_file, 'r', encoding='cp1252', errors='replace') as f:
        content = f.read()

    start_idx = text.find(tag_start)
    end_idx = text.find(tag_end, start_idx)
    
    if start_idx != -1 and end_idx != -1:
        new_text = text[:start_idx + len(tag_start)] + '\n\n' + content + '\n' + text[end_idx:]
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(new_text)

inject('reports/report_skeleton.md', '<!-- Auto-populated from reports/training_summary.md -->', '### 6.2', 'reports/training_summary.md')
inject('reports/report_skeleton.md', '<!-- Auto-populated from reports/training_summary_multimodal.md -->', '### 6.3', 'reports/training_summary_multimodal.md')
