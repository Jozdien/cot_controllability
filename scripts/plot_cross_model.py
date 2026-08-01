import json
import glob
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['font.family'] = 'serif'

def analyze_hillclimb(path, variant_filter=None):
    rows = []
    with open(path) as f:
        for line in f:
            if line.startswith('#') or not line.strip(): continue
            rows.append(json.loads(line))
    if variant_filter:
        rows = [r for r in rows if r['variant'] in variant_filter]
    comp_by_mode = {}
    acc_rows = []
    for r in rows:
        m = r['mode']
        if r['compliant'] is not None:
            comp_by_mode.setdefault(m, []).append(r['compliant'])
        sid = r.get('sample_id', '')
        if not sid and 'sample' in r and isinstance(r['sample'], dict):
            sid = r['sample'].get('id', '')
        if r.get('correct') is not None and not sid.startswith('cotcontrol/mmlu_pro'):
            acc_rows.append(r['correct'])
    mode_rates = [sum(v)/len(v) for v in comp_by_mode.values() if len(v) > 0]
    macro_comp = sum(mode_rates)/len(mode_rates) if mode_rates else 0
    acc = sum(acc_rows)/len(acc_rows) if acc_rows else 0
    return macro_comp * 100, acc * 100

def analyze_fewshot(dirpath):
    rows = []
    for f in glob.glob(f'{dirpath}/*.jsonl'):
        with open(f) as fh:
            for line in fh:
                if line.startswith('#') or not line.strip(): continue
                try: rows.append(json.loads(line))
                except: continue
    comp_by_mode = {}
    acc_rows = []
    for r in rows:
        m = r.get('control_mode', r.get('mode', ''))
        if r.get('compliant') is not None:
            comp_by_mode.setdefault(m, []).append(r['compliant'])
        sid = r.get('sample_id', '')
        if not sid and 'sample' in r and isinstance(r['sample'], dict):
            sid = r['sample'].get('id', '')
        if r.get('correct') is not None and not sid.startswith('cotcontrol/mmlu_pro'):
            acc_rows.append(r['correct'])
    mode_rates = [sum(v)/len(v) for v in comp_by_mode.values() if len(v) > 0]
    macro_comp = sum(mode_rates)/len(mode_rates) if mode_rates else 0
    acc = sum(acc_rows)/len(acc_rows) if acc_rows else 0
    return macro_comp * 100, acc * 100

R = 'results'

# Gather data: (model, config_label, compliance, accuracy)
data = []

# Qwen3-32B
configs_32b = [
    ('ZS baseline', lambda: analyze_hillclimb(f'{R}/full_eval_all_qwen/hillclimb_qwen3-32b.jsonl', ['baseline'])),
    ('ZS metacognition', lambda: analyze_hillclimb(f'{R}/full_eval_all_qwen/hillclimb_qwen3-32b.jsonl', ['metacognition'])),
    ('ZS repeat_15x', lambda: analyze_hillclimb(f'{R}/full_eval_all_zs7/hillclimb_qwen3-32b.jsonl', ['repeat_15x'])),
    ('ZS all_weapons_nopf', lambda: analyze_hillclimb(f'{R}/full_eval_all_zs8/hillclimb_qwen3-32b.jsonl', ['all_weapons_noprefill'])),
    ('ZS prefill_only', lambda: analyze_hillclimb(f'{R}/full_eval_all_zs7/hillclimb_qwen3-32b.jsonl', ['prefill_only'])),
    ('ZS prefill+repeat', lambda: analyze_hillclimb(f'{R}/full_eval_all_zs7/hillclimb_qwen3-32b.jsonl', ['prefill+repeat'])),
    ('ZS all_weapons', lambda: analyze_hillclimb(f'{R}/full_eval_all_zs7/hillclimb_qwen3-32b.jsonl', ['all_weapons'])),
    ('FS OOD 5s', lambda: analyze_fewshot(f'{R}/full_eval_all_ood5s')),
    ('FS OOD 5s+in+pf', lambda: analyze_fewshot(f'{R}/full_eval_all_ood5s_inpf')),
    ('FS 1000c/10s', lambda: analyze_fewshot(f'{R}/full_eval_all_1000c10s')),
    ('FS 1500c/10s', lambda: analyze_fewshot(f'{R}/full_eval_all_1500c10s')),
]
for label, fn in configs_32b:
    comp, acc = fn()
    data.append(('Qwen3-32B', label, comp, acc))

# GPT-OSS-120B
configs_120b = [
    ('ZS baseline', lambda: analyze_hillclimb(f'{R}/full_eval_all_qwen_oss120b/hillclimb_gpt-oss-120b.jsonl', ['baseline'])),
    ('ZS metacognition', lambda: analyze_hillclimb(f'{R}/full_eval_all_qwen_oss120b/hillclimb_gpt-oss-120b.jsonl', ['metacognition'])),
    ('ZS repeat_15x', lambda: analyze_hillclimb(f'{R}/full_eval_all_zs7_oss120b/hillclimb_gpt-oss-120b.jsonl', ['repeat_15x'])),
    ('ZS all_weapons_nopf', lambda: analyze_hillclimb(f'{R}/full_eval_all_zs8_oss120b/hillclimb_gpt-oss-120b.jsonl', ['all_weapons_noprefill'])),
    ('ZS prefill_only', lambda: analyze_hillclimb(f'{R}/full_eval_all_zs7_oss120b/hillclimb_gpt-oss-120b.jsonl', ['prefill_only'])),
    ('ZS prefill+repeat', lambda: analyze_hillclimb(f'{R}/full_eval_all_zs7_oss120b/hillclimb_gpt-oss-120b.jsonl', ['prefill+repeat'])),
    ('ZS all_weapons', lambda: analyze_hillclimb(f'{R}/full_eval_all_zs7_oss120b/hillclimb_gpt-oss-120b.jsonl', ['all_weapons'])),
    ('FS OOD 5s', lambda: analyze_fewshot(f'{R}/full_eval_all_ood5s_oss120b')),
    ('FS OOD 5s+in+pf', lambda: analyze_fewshot(f'{R}/full_eval_all_ood5s_inpf_oss120b')),
    ('FS 1000c/10s', lambda: analyze_fewshot(f'{R}/full_eval_all_1000c10s_oss120b')),
]
for label, fn in configs_120b:
    comp, acc = fn()
    data.append(('GPT-OSS-120B', label, comp, acc))

# GPT-OSS-20B
configs_20b = [
    ('ZS baseline', lambda: analyze_hillclimb(f'{R}/full_eval_all_qwen_oss20b/hillclimb_gpt-oss-20b.jsonl', ['baseline'])),
    ('ZS metacognition', lambda: analyze_hillclimb(f'{R}/full_eval_all_qwen_oss20b/hillclimb_gpt-oss-20b.jsonl', ['metacognition'])),
    ('ZS repeat_15x', lambda: analyze_hillclimb(f'{R}/full_eval_all_qwen_oss20b/hillclimb_gpt-oss-20b.jsonl', ['repeat_15x'])),
    ('ZS all_weapons_nopf', lambda: analyze_hillclimb(f'{R}/full_eval_all_zs8_oss20b/hillclimb_gpt-oss-20b.jsonl', ['all_weapons_noprefill'])),
    ('ZS prefill_only', lambda: analyze_hillclimb(f'{R}/full_eval_all_zs7_oss20b/hillclimb_gpt-oss-20b.jsonl', ['prefill_only'])),
    ('ZS prefill+repeat', lambda: analyze_hillclimb(f'{R}/full_eval_all_zs7_oss20b/hillclimb_gpt-oss-20b.jsonl', ['prefill+repeat'])),
    ('ZS all_weapons', lambda: analyze_hillclimb(f'{R}/full_eval_all_zs7_oss20b/hillclimb_gpt-oss-20b.jsonl', ['all_weapons'])),
    ('FS OOD 5s', lambda: analyze_fewshot(f'{R}/full_eval_all_ood5s_oss20b')),
    ('FS OOD 5s+in+pf', lambda: analyze_fewshot(f'{R}/full_eval_all_ood5s_inpf_oss20b')),
    ('FS 1000c/10s', lambda: analyze_fewshot(f'{R}/full_eval_all_1000c10s_oss20b')),
]
for label, fn in configs_20b:
    comp, acc = fn()
    data.append(('GPT-OSS-20B', label, comp, acc))

# Qwen3-8B
configs_8b = [
    ('ZS baseline', lambda: analyze_hillclimb(f'{R}/full_eval_all_qwen_8b/hillclimb_qwen3-8b.jsonl', ['baseline'])),
    ('ZS metacognition', lambda: analyze_hillclimb(f'{R}/full_eval_all_qwen_8b/hillclimb_qwen3-8b.jsonl', ['metacognition'])),
    ('ZS repeat_15x', lambda: analyze_hillclimb(f'{R}/full_eval_all_zs7_8b/hillclimb_qwen3-8b.jsonl', ['repeat_15x'])),
    ('ZS all_weapons_nopf', lambda: analyze_hillclimb(f'{R}/full_eval_all_zs8_8b/hillclimb_qwen3-8b.jsonl', ['all_weapons_noprefill'])),
    ('ZS prefill_only', lambda: analyze_hillclimb(f'{R}/full_eval_all_zs7_8b/hillclimb_qwen3-8b.jsonl', ['prefill_only'])),
    ('ZS prefill+repeat', lambda: analyze_hillclimb(f'{R}/full_eval_all_zs7_8b/hillclimb_qwen3-8b.jsonl', ['prefill+repeat'])),
    ('ZS all_weapons', lambda: analyze_hillclimb(f'{R}/full_eval_all_zs7_8b/hillclimb_qwen3-8b.jsonl', ['all_weapons'])),
    ('FS OOD 5s', lambda: analyze_fewshot(f'{R}/full_eval_all_ood5s_8b')),
    ('FS OOD 5s+in+pf', lambda: analyze_fewshot(f'{R}/full_eval_all_ood5s_inpf_8b')),
    ('FS 1000c/10s', lambda: analyze_fewshot(f'{R}/full_eval_all_1000c10s_8b')),
]
for label, fn in configs_8b:
    comp, acc = fn()
    data.append(('Qwen3-8B', label, comp, acc))

# Plot
fig, ax = plt.subplots(figsize=(11, 7))

model_colors = {
    'Qwen3-32B': '#2563eb',
    'Qwen3-8B': '#7c3aed',
    'GPT-OSS-20B': '#dc2626',
    'GPT-OSS-120B': '#ea580c',
}

model_markers = {
    'Qwen3-32B': 'o',
    'Qwen3-8B': 's',
    'GPT-OSS-20B': 'D',
    'GPT-OSS-120B': '^',
}

zs_configs = {'ZS baseline', 'ZS metacognition', 'ZS repeat_15x', 'ZS all_weapons_nopf', 'ZS prefill_only', 'ZS prefill+repeat', 'ZS all_weapons'}

from matplotlib.lines import Line2D

for model, label, comp, acc in data:
    color = model_colors[model]
    marker = model_markers[model]
    is_zs = label in zs_configs
    size = 80

    fc = color if is_zs else 'white'
    ax.scatter(comp, acc, marker=marker, s=size,
               edgecolors=color, linewidths=1.5,
               facecolors=fc, zorder=5, alpha=0.85)

# Connect points per model with thin lines to show progression
for model in model_colors:
    pts = [(comp, acc, label) for m, label, comp, acc in data if m == model]
    pts.sort(key=lambda x: x[0])
    if len(pts) > 1:
        ax.plot([p[0] for p in pts], [p[1] for p in pts],
                color=model_colors[model], alpha=0.15, linewidth=1, zorder=2)

# Annotate key points only
annotate_points = {
    ('Qwen3-32B', 'ZS baseline'): (8, 3),
    ('Qwen3-32B', 'ZS all_weapons_nopf'): (5, 4),
    ('Qwen3-32B', 'ZS all_weapons'): (-2, -12),
    ('Qwen3-32B', 'FS OOD 5s+in+pf'): (5, -10),
    ('Qwen3-32B', 'FS 1000c/10s'): (5, 4),
    ('GPT-OSS-120B', 'ZS baseline'): (6, -8),
    ('GPT-OSS-120B', 'ZS all_weapons_nopf'): (5, -10),
    ('GPT-OSS-120B', 'ZS all_weapons'): (5, 4),
    ('GPT-OSS-120B', 'FS 1000c/10s'): (5, -10),
    ('GPT-OSS-20B', 'ZS baseline'): (-5, -12),
    ('GPT-OSS-20B', 'ZS all_weapons_nopf'): (5, -10),
    ('GPT-OSS-20B', 'ZS all_weapons'): (5, 4),
    ('GPT-OSS-20B', 'FS 1000c/10s'): (5, -10),
    ('Qwen3-8B', 'ZS baseline'): (8, 3),
    ('Qwen3-8B', 'ZS all_weapons_nopf'): (5, -10),
    ('Qwen3-8B', 'ZS all_weapons'): (5, 4),
    ('Qwen3-8B', 'FS 1000c/10s'): (5, -10),
}
for model, label, comp, acc in data:
    key = (model, label)
    if key in annotate_points:
        dx, dy = annotate_points[key]
        short = label.replace('ZS ', '').replace('FS ', '')
        ax.annotate(short, (comp, acc),
                    textcoords='offset points', xytext=(dx, dy),
                    fontsize=7, color=model_colors[model], alpha=0.75,
                    fontweight='bold')

# Legend: models
model_handles = [Line2D([0], [0], marker=model_markers[m], color='w',
                        markerfacecolor=c, markersize=9, label=m, markeredgecolor=c)
                 for m, c in model_colors.items()]

# Legend: ZS vs FS
style_handles = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
           markeredgecolor='gray', markersize=8, label='Zero-shot'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
           markeredgecolor='gray', markersize=8, label='Few-shot', markeredgewidth=1.5),
]

leg1 = ax.legend(handles=model_handles, loc='upper right', title='Model', fontsize=9, title_fontsize=10)
ax.add_artist(leg1)
ax.legend(handles=style_handles, loc='lower right', title='Style', fontsize=9, title_fontsize=10)

ax.set_xlabel('Macro-Averaged Compliance (%)', fontsize=12)
ax.set_ylabel('Accuracy (%, excl. MMLU_PRO)', fontsize=12)
ax.set_title('CoT Controllability: Accuracy vs Compliance Tradeoff', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.2)
ax.set_xlim(-1, 38)
ax.set_ylim(18, 42)

ax.annotate('ZS accuracy inflated ~5pp\ndue to error exclusion',
            xy=(0.02, 0.02), xycoords='axes fraction', fontsize=7,
            fontstyle='italic', color='gray')

plt.tight_layout()
plt.savefig('results/plots/cross_model_acc_vs_comp.png', dpi=200, bbox_inches='tight')
plt.savefig('results/plots/cross_model_acc_vs_comp.pdf', bbox_inches='tight')
print('Saved to results/plots/cross_model_acc_vs_comp.png')
