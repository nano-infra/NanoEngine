#!/usr/bin/env python3
"""Regenerate the Chapter 8 measurement tables and figure from saved per-rank CSVs."""
from __future__ import annotations
import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / 'bench/k3_partition/results'
DOC = ROOT / 'docs/blogs/dlengine-kda-evaluation.md'


def read(directory, file):
    return list(csv.DictReader((RESULTS / directory / (file + '.csv')).open()))


def maximum(rows, column, **where):
    selected = [float(row[column]) for row in rows if all(row[k] == str(v) for k, v in where.items())]
    if not selected:
        raise ValueError((column, where))
    return max(selected)


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |']
                     + ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def main():
    kda = read('gb200_graph_kda', 'kda')
    cp = read('gb200_graph_cp', 'cp')
    moe = read('gb200_graph_moe', 'moe')
    comm = read('gb200', 'collectives')
    transition_path = RESULTS / 'gb200_transition/transition.csv'
    transition = read('gb200_transition', 'transition') if transition_path.exists() else []
    blocks = {}
    blocks['KDA_MEASUREMENT_TABLE'] = table(
        ['TP', 'DP', '8K Prefill, eager (ms)', 'Decode B=1, graph (µs)', 'Decode B=128, graph (µs)', 'Decode B=256, graph (µs)'],
        [[tp, 16 // tp,
          f"{maximum(kda, 'ms', tp=tp, phase='prefill', chunk=8192):.3f}",
          *[f"{1000 * maximum(kda, 'graph_ms', tp=tp, phase='decode', batch_per_dp=b):.1f}" for b in [1, 128, 256]]]
         for tp in [1, 2, 4, 8, 16]])
    blocks['CP_MEASUREMENT_TABLE'] = table(
        ['Effective head TP', 'CP', '32K cached tokens (ms)', '128K cached tokens (ms)', '1M cached tokens (ms)'],
        [[tp, p, *[f"{maximum(cp, 'graph_ms', tp=tp, cp=p, queries=8192, cached_tokens=l):.3f}" for l in [32768, 131072, 1048576]]]
         for tp, p in [(16, 1), (8, 2), (4, 4), (2, 8), (1, 16)]])
    blocks['MOE_MEASUREMENT_TABLE'] = table(
        ['EP16 total input rows', 'Balanced graph (ms)', 'Hot graph (ms)'],
        [[n, *[f"{maximum(moe, 'graph_ms', ep=16, source_ranks=16, global_tokens=n, routing=r):.3f}" for r in ['balanced', 'hot_rank']]]
         for n in [16, 128, 2048, 8192, 32768]])
    blocks['SOURCE_MEASUREMENT_TABLE'] = table(
        ['Source ranks', 'Rows/active source', 'Routed path, graph (ms)'],
        [[s, 8192 // s, f"{maximum(moe, 'graph_ms', ep=16, source_ranks=s, global_tokens=8192, routing='balanced'):.3f}"] for s in [4, 8, 16]])
    composition = []
    for tp in [4, 8, 16]:
        k = maximum(kda, 'ms', tp=tp, phase='prefill', chunk=8192)
        ar = maximum(comm, 'median_ms', size=tp, requested_tokens=8192, op='all_reduce')
        pair = maximum(comm, 'median_ms', size=tp, requested_tokens=8192, op='rs_ag_pair')
        a = maximum(cp, 'attention_with_merge_ms', tp=tp, cp=1, queries=8192, cached_tokens=1048576)
        f = maximum(moe, 'ms', ep=16, source_ranks=tp, global_tokens=8192, routing='balanced')
        values = [69 * (k - ar), 24 * a, 92 * f, 93 * pair]
        composition.append([f'DP{16 // tp}×TP{tp}/EP16', *[f'{v / 1000:.3f}' for v in values], f'{sum(values) / 1000:.3f}'])
    blocks['COMPOSITION_TABLE'] = table(
        ['Layout', '69× KDA minus AR (s)', '24× MLA prefix core (s)', '92× routed MoE (s)', '93× RS→AG (s)', 'Measured-boundary subtotal (s)'], composition)
    if transition:
        blocks['TRANSITION_MEASUREMENT_TABLE'] = table(
            ['Total rows', 'Original sources', 'Original routed path (ms)', 'Redistribution + routed path + inverse (ms)', 'Redistribution round trip alone (ms)'],
            [[n, a, *[f"{maximum(transition, 'median_ms', global_tokens=n, source_ranks=a, operation=op):.3f}" for op in ['original_owners', 'redistribute_routed_restore', 'redistribute_restore_only']]] for n in [2048, 8192] for a in [4, 8]])
    doc = DOC.read_text()
    for name, content in blocks.items():
        block = '<!-- BEGIN ' + name + ' -->\n' + content + '\n<!-- END ' + name + ' -->'
        doc = doc.replace('<!-- ' + name + ' -->', block)
        doc = re.sub(r'<!-- BEGIN ' + name + r' -->.*?<!-- END ' + name + r' -->', lambda _: block, doc, flags=re.S)
    DOC.write_text(doc)
    summary = dict(kda_rows=len(kda), cp_rows=len(cp), moe_rows=len(moe),
                   cp_max_abs=max(float(x['correctness_max_abs']) for x in cp),
                   transition_rows=len(transition), composition=composition)
    (RESULTS / 'gb200/summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    plot(kda, cp, moe, transition)


def plot(kda, cp, moe, transition):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({'font.size': 10, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    colors = ['#475569', '#2563eb', '#0891b2', '#16a34a', '#d97706']
    model = json.loads((RESULTS / 'gb200/joint_model.json').read_text())
    layouts = model['layout_estimates'][:7]
    names = ['TP1', 'TP2', 'TP4', 'TP8', 'TP16', 'TP8\nCP2*', 'TP8\nCP4*']
    ax = axes[0, 0]
    weight = np.array([x['weights_gib'] for x in layouts])
    ax.bar(names, weight, color='#475569', label='Estimated weights')
    ax.bar(names, [20] * len(names), bottom=weight, color='#cbd5e1', label='20 GiB planning allowance')
    ax.axhline(model['hbm_bytes'] / 2**30, color='#dc2626', ls='--', label='Measured HBM capacity')
    ax.set_ylabel('GiB / GPU'); ax.set_title('(a) EP16 weight fit; *nested CP is a design estimate')
    ax.legend(fontsize=8, loc='lower right')
    ax = axes[0, 1]
    batches = [1, 8, 32, 128, 256]
    for tp, color in zip([1, 2, 4, 8, 16], colors):
        y = [maximum(kda, 'graph_ms', tp=tp, phase='decode', batch_per_dp=b) * 1000 for b in batches]
        ax.plot(batches, y, 'o-', color=color, label=f'TP{tp}')
    ax.set_xscale('log', base=2); ax.set_xlabel('Decode sequences / DP group')
    ax.set_ylabel('Complete KDA graph latency (µs)'); ax.set_title('(b) KDA: TP gains flatten at small batches')
    ax.legend(ncol=3, fontsize=8); ax.grid(alpha=.2)
    ax = axes[1, 0]
    lengths = [32768, 131072, 1048576]
    for (tp, cp_degree), color in zip([(16, 1), (8, 2), (4, 4), (2, 8), (1, 16)], colors):
        y = [maximum(cp, 'graph_ms', tp=tp, cp=cp_degree, queries=8192, cached_tokens=l) for l in lengths]
        ax.plot([l / 1024 for l in lengths], y, 'o-', color=color, label=f'head TP{tp}, CP{cp_degree}')
    ax.set_xscale('log', base=2); ax.set_yscale('log')
    ax.set_xlabel('Cached prefix (Ki tokens), 8K fresh queries'); ax.set_ylabel('Attention + CP merge (ms)')
    ax.set_title('(c) CP: fixed 16 GPUs, expanded KV core only')
    ax.legend(fontsize=8); ax.grid(alpha=.2)
    ax = axes[1, 1]
    positions = np.arange(3)
    sources = [4, 8, 16]
    original = [maximum(moe, 'graph_ms', ep=16, source_ranks=s, global_tokens=8192, routing='balanced') for s in sources]
    if transition:
        original = [maximum(transition, 'median_ms', global_tokens=8192, source_ranks=s, operation='original_owners') for s in sources]
        changed = [maximum(transition, 'median_ms', global_tokens=8192, source_ranks=s, operation='redistribute_routed_restore') for s in sources]
        ax.bar(positions - .18, original, width=.36, color='#2563eb', label='Original owners')
        ax.bar(positions + .18, changed, width=.36, color='#16a34a', label='Redistribute + MoE + inverse')
    else:
        ax.bar(positions, original, color='#2563eb', label='Original owners')
    ax.set_xticks(positions, sources); ax.set_xlabel('Original source ranks (8K tokens, same expert loads)')
    ax.set_ylabel('Routed path graph latency (ms)'); ax.set_title('(d) FFN input ownership changes performance')
    ax.legend(fontsize=8)
    fig.suptitle('K3 parallel-layout evidence on 16 GB200 GPUs\nComponent measurements and capacity estimates; not full-model serving benchmarks', fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .94))
    destination = ROOT / 'docs/blogs/assets/k3-gb200-joint.svg'
    fig.savefig(destination, bbox_inches='tight')
    fig.savefig('/tmp/k3-gb200-joint.png', dpi=140, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
