# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Compare identical baseline and candidate HTTP workloads using saved client results."""
import argparse
import json
from pathlib import Path


def load(directory, artifact):
    return json.loads((directory / artifact).read_text())


def performance(baseline, candidate):
    assert baseline.keys() == candidate.keys()
    by_case = {}
    for case, first in baseline.items():
        second = candidate[case]
        a_cache = [row['cached_tokens'] for row in first['rounds']]
        b_cache = [row['cached_tokens'] for row in second['rounds']]
        metrics = {}
        for key in ('median_ttft_ms', 'median_tok_s', 'median_tpot_ms'):
            a, b = first[key], second[key]
            metrics[key] = {'baseline': a, 'candidate': b, 'delta_pct': 100 * (b / a - 1)}
        by_case[case] = {'matching_cache_hits': a_cache == b_cache, 'baseline_cache': a_cache,
                         'candidate_cache': b_cache, 'metrics': metrics}
    return by_case


def accuracy(baseline, candidate):
    assert baseline['indices'] == candidate['indices']
    assert baseline['count'] == candidate['count'] == len(baseline['indices'])
    diffs = []
    summary = {'count': baseline['count'], 'baseline_correct': baseline['correct'],
               'candidate_correct': candidate['correct'], 'same_output_ids': 0, 'same_accept_counts': 0}
    for a, b in zip(baseline['rows'], candidate['rows']):
        assert (a['dataset_index'], a['input'], a['expected']) == (b['dataset_index'], b['input'], b['expected'])
        ar, br = a['response'], b['response']
        same_ids = ar['output_ids'] == br['output_ids']
        a_meta, b_meta = ar['meta_info'], br['meta_info']
        a_stats = {k: a_meta.get(k) for k in ('spec_verify_ct', 'spec_num_correct_drafts', 'spec_num_proposed_drafts')}
        b_stats = {k: b_meta.get(k) for k in a_stats}
        same_stats = a_stats == b_stats and a_stats['spec_num_correct_drafts'] is not None
        summary['same_output_ids'] += same_ids
        summary['same_accept_counts'] += same_stats
        if not (same_ids and same_stats and a['correct'] == b['correct']):
            diffs.append({'i': a['dataset_index'], 'same_tokens': same_ids, 'baseline_spec': a_stats,
                          'candidate_spec': b_stats, 'baseline_correct': a['correct'], 'candidate_correct': b['correct']})
    summary['changed_rows'] = diffs
    return summary


def cross_batch(records):
    grouped = {}
    for variant in ('alone', 'reversed'):
        selected = [row for row in records if row['variant'] == variant]
        assert selected, variant
        mismatches = []
        for row in selected:
            counts = row['accept_counts']
            accept_equal = all(a is not None and a == b for a,b in counts.values())
            if not row['tokens_equal'] or not accept_equal:
                mismatches.append({'dataset_index': row['dataset_index'],
                                   'tokens_equal': row['tokens_equal'],
                                   'accept_counts': counts})
        grouped[variant] = {
            'samples': len(selected),
            'identical_output_ids': sum(row['tokens_equal'] for row in selected),
            'identical_accept_counts': sum(
                all(a is not None and a == b for a,b in row['accept_counts'].values())
                for row in selected
            ),
            'mismatches': mismatches,
        }
    return grouped


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    bas, cand = args.baseline, args.candidate
    metrics = {'performance': performance(load(bas, 'performance.json'), load(cand, 'performance.json'))}
    if all((directory / 'gsm8k-results.json').exists() for directory in (bas, cand)):
        metrics['accuracy'] = accuracy(load(bas, 'gsm8k-results.json'), load(cand, 'gsm8k-results.json'))
    if (cand / 'gsm8k-cross-batch.json').exists():
        metrics['cross_batch'] = cross_batch(load(cand, 'gsm8k-cross-batch.json'))
    print(json.dumps(metrics, indent=2))
