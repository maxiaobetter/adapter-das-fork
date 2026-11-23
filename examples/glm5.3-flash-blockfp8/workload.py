# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Real-request, paired call, performance, and answer-comparison inputs for TP8."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
import random
import re
from statistics import median
from time import monotonic, sleep
from urllib.error import HTTPError
from urllib.request import Request, urlopen

SEED = 42
ANCHOR = ('A bakery sold 17 rolls in the morning and 26 in the afternoon. '
          'How many rolls did it sell? Explain briefly and give the total.')
JAPAN = 'What is the capital of Japan? Reply with only the city name.'
FRANCE = 'What is the capital of France? Reply with only the city name.'
SHARED = (
    'Use the information in the question to compute the required quantity. '
    'Use exact arithmetic. Give the final number after the delimiter ####. '
    'For example, when the question asks for the sum of 13 boxes and 8 boxes, '
    'the answer line is #### 21. Use a decimal point when the value is not an '
    'integer. Keep units out of the answer line. If a question describes '
    'several people or times, identify the location, person, or time requested '
    'before doing the calculation. Show enough working to explain what the '
    'terms in the calculation stand for. Your final line is the numeric '
    'answer. Do not append punctuation to the numeric line.\n\n'
)

CACHE_PREFIX = SHARED + (
    'Worked example one. Ren buys three packets of pencils. Each packet contains '
    'twelve pencils. She uses five pencils in her classroom and gives seven pencils '
    'to her neighbor. Multiplying gives 36 pencils at the start. The two uses account '
    'for 12 pencils, so 24 remain. The quantity requested is the count remaining. '
    'The final answer is #### 24.\n\n'
    'Worked example two. A bus leaves a station with eight passengers. At the first '
    'stop, six passengers board and two get off. At the second stop, three board and '
    'none get off. The starting count is eight. The first change adds four, and the '
    'second adds three. The bus then has fifteen passengers. The final answer is '
    '#### 15.\n\n'
    'Worked example three. A shop has ten blue notebooks and twenty red notebooks. '
    'It sells five blue notebooks and four red notebooks. The initial inventory is '
    'thirty and the number sold is nine. Subtract nine from thirty to determine the '
    'remaining inventory, including both colors. The final answer is #### 21.\n\n'
    'Now solve the next question without copying the numbers from the examples.\n\n'
)


def prompt(text):
    return '[gMASK]<sop><|user|>' + text + '<|assistant|>'


def payload(texts, max_tokens, stream=False, logprobs=False, ignore_eos=False):
    return {
        'text': [prompt(t) for t in texts],
        'stream': stream,
        'sampling_params': {'temperature': 0, 'sampling_seed': SEED,
                            'max_new_tokens': max_tokens, 'ignore_eos': ignore_eos},
        'return_logprob': logprobs, 'top_logprobs_num': 3 if logprobs else 0,
        'return_prompt_token_ids': not stream,
    }


class Service:
    def __init__(self, url):
        self.url = url

    def post(self, path, body=None):
        request = Request(self.url + path, data=json.dumps(body or {}).encode(),
                          headers={'Content-Type': 'application/json'}, method='POST')
        return urlopen(request, timeout=300)

    def flush(self):
        deadline = monotonic() + 90
        while True:
            try:
                with self.post('/flush_cache') as response:
                    response.read()
                return
            except HTTPError as error:
                if error.code != 400 or monotonic() >= deadline:
                    raise
                sleep(1)

    def generate(self, texts, max_tokens, **kwargs):
        with self.post('/generate', payload(texts, max_tokens, **kwargs)) as response:
            data = json.load(response)
        assert isinstance(data, list) and len(data) == len(texts), data
        assert all(item['output_ids'] for item in data), data
        return data

    def stream(self, text, max_tokens):
        start = monotonic()
        first = None
        final = None
        last = None
        body = payload([text], max_tokens, stream=True, ignore_eos=True)
        body['text'] = body['text'][0]
        with self.post('/generate', body) as response:
            for raw in response:
                if not raw.startswith(b'data:'):
                    continue
                value = raw[5:].strip()
                if value == b'[DONE]':
                    break
                packet = json.loads(value)
                if 'error' in packet:
                    raise RuntimeError(packet['error'])
                n = packet['meta_info']['completion_tokens']
                if n and first is None:
                    first = monotonic()
                if final is None or n > final['meta_info']['completion_tokens']:
                    last = monotonic()
                final = packet
        assert final and first is not None and last is not None
        assert final['meta_info']['completion_tokens'] == max_tokens
        return {'start': start, 'first': first, 'last': last, 'response': final}


def write(path, result):
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')


def calls(server, output, dataset, capture_graph_buffers):
    results = {}
    cases = [('prefill_decode_b1', [ANCHOR]), ('prefill_decode_b2', [ANCHOR, JAPAN])]
    for case, texts in cases:
        server.flush()
        server.generate(texts, 20, ignore_eos=True)
        server.flush()
        results[case] = {'payload': payload(texts, 20, ignore_eos=True),
                         'responses': server.generate(texts, 20, ignore_eos=True)}
        write(output / 'call-responses.json', results)
    for case, text in (('A', JAPAN), ('B', FRANCE), ('A-repeat', JAPAN)):
        server.flush()
        if capture_graph_buffers:
            (output / 'observe-graph').write_text(case)
        try:
            results['graph-' + case] = {'payload': payload([text], 16, logprobs=True),
                                        'responses': server.generate([text], 16, logprobs=True)}
        finally:
            if capture_graph_buffers:
                (output / 'observe-graph').unlink()
        write(output / 'call-responses.json', results)

    if dataset is not None:
        questions = [json.loads(line)['question'] for line in dataset.read_text().splitlines()[:36]]
        exhibit = '\n\n'.join(f'Problem {i + 1}: {question}' for i, question in enumerate(questions))
        query = 'Read the following collection. Solve only problem 8; give its numerical answer.\n\n'
        texts = [query + exhibit]
        server.flush()
        server.generate(texts, 20, ignore_eos=True)
        server.flush()
        results['prefill_decode_long'] = {'payload': payload(texts, 20, ignore_eos=True),
                                          'responses': server.generate(texts, 20, ignore_eos=True)}
        write(output / 'call-responses.json', results)


def performance(server, output, concurrencies, repeats, output_tokens):
    prompts = [CACHE_PREFIX + x for x in (ANCHOR, JAPAN, FRANCE, 'What is five times six?')]
    stats = {}
    for count in concurrencies:
        for warm_cache in (False, True):
            label = f'c{count}-' + ('cached' if warm_cache else 'cold')
            results = []
            with ThreadPoolExecutor(max_workers=count) as workers:
                for repeat in range(repeats):
                    server.flush()
                    if warm_cache:
                        server.generate([CACHE_PREFIX], 1)
                    rows = list(workers.map(lambda x: server.stream(x, output_tokens), prompts[:count]))
                    if repeat == 0:
                        continue
                    elapsed = max(x['last'] for x in rows) - min(x['start'] for x in rows)
                    results.append({'tok_s': output_tokens * count / elapsed,
                                    'ttft_ms': [1000 * (x['first'] - x['start']) for x in rows],
                                    'tpot_ms': [1000 * (x['last'] - x['first']) / (output_tokens - 1) for x in rows],
                                    'cached_tokens': [x['response']['meta_info']['cached_tokens'] for x in rows],
                                    'responses': [x['response'] for x in rows]})
            if warm_cache:
                assert all(all(x > 0 for x in row['cached_tokens']) for row in results), results
            else:
                assert all(min(row['cached_tokens']) == 0 for row in results), results
            stats[label] = {'output_tokens_per_request': output_tokens,
                            'median_tok_s': median(row['tok_s'] for row in results),
                            'median_ttft_ms': median(x for row in results for x in row['ttft_ms']),
                            'median_tpot_ms': median(x for row in results for x in row['tpot_ms']),
                            'rounds': results}
            write(output / 'performance.json', stats)


ANSWER = re.compile(r'####\s*([-+]?\$?\d[\d,]*(?:\.\d+)?)\b')


def numeric(value):
    matches = ANSWER.findall(value)
    if not matches:
        return None
    try:
        return str(Decimal(matches[-1].replace(',', '').replace('$', '')).normalize())
    except InvalidOperation:
        return None


def examples(dataset, sample_size):
    rows = [json.loads(x) for x in dataset.read_text().splitlines()]
    for index in sorted(random.Random(SEED).sample(range(len(rows)), sample_size)):
        row = rows[index]
        expected = numeric(row['answer'])
        assert expected is not None, (index, row)
        yield {'dataset_index': index, 'input': SHARED + row['question'], 'expected': expected}


def accuracy(server, output, dataset, sample_size):
    sample = list(examples(dataset, sample_size))
    results = []
    for index in range(0, len(sample), 2):
        batch = sample[index:index + 2]
        texts = [entry['input'] for entry in batch]
        server.flush()
        responses = server.generate(texts, 512)
        for entry, response in zip(batch, responses):
            answer = numeric(response['text'].rsplit('</think>', 1)[-1])
            results.append({**entry, 'answer': answer, 'correct': answer == entry['expected'], 'response': response})
        write(output / 'gsm8k-results.json', {'count': len(results), 'correct': sum(r['correct'] for r in results),
                                             'indices': [entry['dataset_index'] for entry in sample], 'rows': results})


def cross_batch(server, output):
    reference = json.loads((output / 'gsm8k-results.json').read_text())['rows']
    results = []
    for i in range(0, len(reference) - 1, 8):
        first, second = reference[i:i + 2]
        for name, examples in (('alone', [first]), ('reversed', [second, first])):
            server.flush()
            body = payload([], 512)
            body.pop('text')
            body['input_ids'] = [ex['response']['prompt_token_ids'] for ex in examples]
            with server.post('/generate', body) as raw:
                rows = json.load(raw)
            assert len(rows) == len(examples)
            assert all(row['meta_info']['cached_tokens'] == 0 for row in rows)
            for ex, row in zip(examples, rows):
                expected = ex['response']
                comparisons = {
                    key: [expected['meta_info'].get(key), row['meta_info'].get(key)]
                    for key in ('spec_verify_ct', 'spec_num_correct_drafts', 'spec_num_proposed_drafts')
                }
                results.append({'variant': name, 'dataset_index': ex['dataset_index'],
                                'tokens_equal': expected['output_ids'] == row['output_ids'],
                                'accept_counts': comparisons, 'response': row})
        write(output / 'gsm8k-cross-batch.json', results)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['calls', 'performance', 'accuracy', 'cross-batch'])
    parser.add_argument('output', type=Path)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--dataset', type=Path)
    parser.add_argument('--capture-graph-buffers', action='store_true')
    parser.add_argument('--require-deterministic', action='store_true')
    parser.add_argument('--sample-size', type=int, default=64)
    parser.add_argument('--concurrencies', type=int, nargs='+', choices=(1, 2, 4), default=(1, 2, 4))
    parser.add_argument('--perf-repetitions', type=int, default=5)
    parser.add_argument('--perf-output-tokens', type=int, default=32)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    server = Service(f'http://127.0.0.1:{args.port}')
    if args.stage == 'calls':
        calls(server, args.output, args.dataset, args.capture_graph_buffers)
    elif args.stage == 'performance':
        assert args.perf_repetitions >= 2 and args.perf_output_tokens >= 2
        performance(server, args.output, args.concurrencies, args.perf_repetitions, args.perf_output_tokens)
    elif args.stage == 'accuracy':
        assert args.dataset is not None
        accuracy(server, args.output, args.dataset, args.sample_size)
    else:
        comparisons = cross_batch(server, args.output)
        if args.require_deterministic and (
            not comparisons or any(
                not row['tokens_equal'] or any(
                    a is None or a != b for a, b in row['accept_counts'].values()
                )
                for row in comparisons
            )
        ):
            raise SystemExit('Cross-batch output tokens or MTP counters differ; see gsm8k-cross-batch.json')


if __name__ == '__main__':
    main()
