# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Record all real prefill and EAGLE shapes in one profiler window."""
import argparse
from pathlib import Path
import json
from time import monotonic, sleep

from workload import Service, payload, write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--arm', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output-tokens', type=int, default=8)
    parser.add_argument('--warmup', choices=['all', 'long', 'none'], default='all')
    parser.add_argument('--output-name', default='torch-graph-continuous')
    parser.add_argument('--external-profiler', action='store_true')
    parser.add_argument('--no-recovery', action='store_true')
    args = parser.parse_args()
    output = args.arm / args.output_name
    output.mkdir()
    service = Service(f'http://127.0.0.1:{args.port}')
    reference = json.loads(args.reference.read_text())
    requests = {}
    for case, info in reference.items():
        if case not in ('prefill_decode_b1', 'prefill_decode_b2', 'prefill_decode_long'):
            continue
        ids = [row['prompt_token_ids'] for row in info['responses']]
        request = payload([], args.output_tokens, ignore_eos=True)
        request.pop('text')
        request['input_ids'] = ids[0] if len(ids) == 1 else ids
        requests[case] = request
    assert [sum(map(len, [p['input_ids']] if isinstance(p['input_ids'][0], int) else p['input_ids']))
            for p in requests.values()] == [34, 52, 2113]
    write(output / 'requests.json', requests)
    responses = {}
    started = False
    try:
        for timing in ('warmup', 'profiler-active'):
            if timing == 'profiler-active' and not args.external_profiler:
                with service.post('/start_profile', {
                    'output_dir': str(output), 'activities': ['CPU', 'GPU'],
                    'merge_profiles': False, 'with_stack': False, 'record_shapes': True,
                }) as response:
                    (output / 'profile-start.txt').write_bytes(response.read())
                started = True
            responses[timing] = {}
            for name, request in requests.items():
                if timing == 'warmup' and (
                    args.warmup == 'none' or args.warmup == 'long' and name != 'prefill_decode_long'
                ):
                    continue
                service.flush()
                before = monotonic()
                with service.post('/generate', request) as response:
                    data = json.load(response)
                elapsed = monotonic() - before
                responses[timing][name] = {'wall_seconds': elapsed, 'http_response': data}
                write(output / 'responses.json', responses)
                print(timing, name, f'{elapsed:.3f} s', flush=True)
    finally:
        if started:
            with service.post('/stop_profile') as response:
                (output / 'profile-stop.txt').write_bytes(response.read())

    if args.external_profiler:
        return
    deadline = monotonic() + 45
    while len(list(output.glob('*.trace.json.gz'))) < 8 and monotonic() < deadline:
        sleep(1)
    assert len(list(output.glob('*.trace.json.gz'))) == 8
    if args.no_recovery:
        return
    service.flush()
    with service.post('/generate', requests['prefill_decode_b1']) as response:
        (output / 'recovery.json').write_bytes(response.read())


if __name__ == '__main__':
    main()
