import json
import glob
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['font.family'] = 'serif'

def _extract_sid(r):
    sid = r.get('sample_id', '')
    if not sid and 'sample' in r and isinstance(r['sample'], dict):
        sid = r['sample'].get('id', '')
    return sid

def _compute_stats(comp_by_mode, acc_rows):
    mode_stats = []
    for v in comp_by_mode.values():
        if not v:
            continue
        n = len(v)
        p = sum(v) / n
        se = np.sqrt(p * (1 - p) / n) if n > 0 else 0
        mode_stats.append({"rate": p, "se": se})
    if not mode_stats:
        return 0, 0, 0, 0
    macro_comp = np.mean([s["rate"] for s in mode_stats])
    comp_se = np.sqrt(np.sum([s["se"] ** 2 for s in mode_stats])) / len(mode_stats)
    acc = sum(acc_rows) / len(acc_rows) if acc_rows else 0
    n = len(acc_rows)
    acc_se = np.sqrt(acc * (1 - acc) / n) if n > 0 else 0
    return macro_comp * 100, comp_se * 100, acc * 100, acc_se * 100

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
        sid = _extract_sid(r)
        if r.get('correct') is not None and not sid.startswith('cotcontrol/mmlu_pro'):
            acc_rows.append(r['correct'])
    return _compute_stats(comp_by_mode, acc_rows)

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
        sid = _extract_sid(r)
        if r.get('correct') is not None and not sid.startswith('cotcontrol/mmlu_pro'):
            acc_rows.append(r['correct'])
    return _compute_stats(comp_by_mode, acc_rows)

R = 'results'

config_order = [
    'ZS baseline', 'ZS metacognition', 'ZS repeat_15x',
    'ZS all_weapons_nopf', 'ZS prefill_only', 'ZS prefill+repeat', 'ZS all_weapons',
    'FS OOD 5s', 'FS OOD 5s+in+pf', 'FS 1000c/10s',
]

config_short = {
    'ZS baseline': 'Baseline',
    'ZS metacognition': 'Meta-\ncognition',
    'ZS repeat_15x': 'Repeat\n15×',
    'ZS all_weapons_nopf': 'All Weapons\n(no PF)',
    'ZS prefill_only': 'Prefill',
    'ZS prefill+repeat': 'Prefill+\nRepeat',
    'ZS all_weapons': 'All\nWeapons',
    'FS OOD 5s': 'OOD\n5-shot',
    'FS OOD 5s+in+pf': 'OOD 5s+\nInline+PF',
    'FS 1000c/10s': 'On-policy\n1000c/10s',
}

model_configs = {
    'Qwen3-8B': [
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
    ],
    'Qwen3-32B': [
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
    ],
    'GPT-OSS-20B': [
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
    ],
    'GPT-OSS-120B': [
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
    ],
}

# Collect results: (comp, comp_se, acc, acc_se)
results = {}
for model_name, cfgs in model_configs.items():
    results[model_name] = {}
    for label, fn in cfgs:
        try:
            results[model_name][label] = fn()
        except Exception:
            pass

model_order = ['Qwen3-8B', 'Qwen3-32B', 'GPT-OSS-20B', 'GPT-OSS-120B']
model_colors = {
    'Qwen3-8B': '#7c3aed',
    'Qwen3-32B': '#2563eb',
    'GPT-OSS-20B': '#dc2626',
    'GPT-OSS-120B': '#ea580c',
}

n_configs = len(config_order)
n_models = len(model_order)

fig, (ax_comp, ax_acc) = plt.subplots(2, 1, figsize=(20, 10.5), sharex=True)

bar_width = 0.18
group_width = n_models * bar_width + 0.15
x_group = np.arange(n_configs) * group_width

for i, model in enumerate(model_order):
    comps, comp_errs, accs, acc_errs = [], [], [], []
    for cfg in config_order:
        r = results.get(model, {}).get(cfg)
        if r:
            comps.append(r[0]); comp_errs.append(r[1] * 1.96)
            accs.append(r[2]); acc_errs.append(r[3] * 1.96)
        else:
            comps.append(0); comp_errs.append(0)
            accs.append(0); acc_errs.append(0)

    offset = (i - (n_models - 1) / 2) * bar_width
    x_pos = x_group + offset
    bw = bar_width * 0.88

    ax_comp.bar(x_pos, comps, bw, yerr=comp_errs, label=model,
                color=model_colors[model], alpha=0.85, edgecolor='white', linewidth=0.5,
                capsize=2, error_kw={'linewidth': 0.8, 'alpha': 0.6})
    ax_acc.bar(x_pos, accs, bw, yerr=acc_errs, label=model,
               color=model_colors[model], alpha=0.85, edgecolor='white', linewidth=0.5,
               capsize=2, error_kw={'linewidth': 0.8, 'alpha': 0.6})

    for j, (x, c, ce, a, ae) in enumerate(zip(x_pos, comps, comp_errs, accs, acc_errs)):
        if c > 0:
            ax_comp.text(x, c + ce + 0.5, f'{c:.0f}', ha='center', va='bottom',
                         fontsize=5.5, color=model_colors[model], fontweight='bold')
        if a > 0:
            ax_acc.text(x, a + ae + 0.3, f'{a:.0f}', ha='center', va='bottom',
                        fontsize=5.5, color=model_colors[model], fontweight='bold')

ax_comp.set_ylabel('Macro-Avg Compliance (%)', fontsize=12)
ax_comp.set_title('CoT Controllability: Cross-Model Comparison', fontsize=14, fontweight='bold')
ax_comp.legend(fontsize=10, ncol=4, loc='upper left')
ax_comp.grid(axis='y', alpha=0.2)
ax_comp.set_ylim(0, 44)

ax_acc.set_ylabel('Accuracy (%, excl. MMLU_PRO)', fontsize=12)
ax_acc.grid(axis='y', alpha=0.2)
ax_acc.set_ylim(0, 48)

ax_acc.set_xticks(x_group)
ax_acc.set_xticklabels([config_short[c] for c in config_order], fontsize=9, ha='center')

sep_x = (x_group[6] + x_group[7]) / 2
for ax in (ax_comp, ax_acc):
    ax.axvline(sep_x, color='gray', linestyle='--', alpha=0.3, linewidth=1)

sep_pf = (x_group[3] + x_group[4]) / 2
for ax in (ax_comp, ax_acc):
    ax.axvline(sep_pf, color='gray', linestyle=':', alpha=0.2, linewidth=1)

ax_comp.text((x_group[0] + x_group[3]) / 2, ax_comp.get_ylim()[1] * 0.96, 'No Prefill',
             ha='center', fontsize=9, fontstyle='italic', color='gray')
ax_comp.text((x_group[4] + x_group[6]) / 2, ax_comp.get_ylim()[1] * 0.96, 'With Prefill',
             ha='center', fontsize=9, fontstyle='italic', color='gray')
ax_comp.text(x_group[8], ax_comp.get_ylim()[1] * 0.96, 'Few-Shot',
             ha='center', fontsize=10, fontstyle='italic', color='gray')

ax_comp.text(0.99, 0.97, 'Error bars: 95% CI\n(propagated binomial SE)',
             transform=ax_comp.transAxes, fontsize=7, va='top', ha='right',
             fontstyle='italic', color='gray')

ax_acc.annotate('ZS accuracy inflated ~5pp due to error exclusion',
                xy=(0.01, 0.02), xycoords='axes fraction', fontsize=7,
                fontstyle='italic', color='gray')

plt.tight_layout()
plt.savefig('results/plots/cross_model_bars.png', dpi=200, bbox_inches='tight')
plt.savefig('results/plots/cross_model_bars.pdf', bbox_inches='tight')
print('Saved to results/plots/cross_model_bars.png')
