# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Judge the model suite's G3 gate against canned chat completions, without a model server.

tests/tools/run_model_suite.sh runs only in L2 (nightly, real models), so its verdict logic had
no test of its own. The nightly run of 2026-11-25 failed G3 on every model, and the result file
did not say why: with 16 / 8 tokens every reply of a thinking model ended inside <think>.
The gate's functions run here unchanged; only curl is replaced, by a function that answers from
a table and records what it was asked.
"""

import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


ROOT = Path(__file__).resolve().parents[2]
SUITE = ROOT / "tests/tools/run_model_suite.sh"
pytestmark = pytest.mark.skipif(os.name != "posix", reason="Linux Bash gate")

CAPITAL = "What is the capital of France? Answer in one word."
SUM = "What is 3 plus 4? Answer with only the number."


def reply(content, tokens=10, finish="stop"):
    return json.dumps({"choices": [{"message": {"role": "assistant", "content": content},
                                    "finish_reason": finish}],
                       "usage": {"completion_tokens": tokens}})


THINKING_CAPITAL = "<think>\nOkay, the user is asking for the capital of France.\n</think>\n\nParis"
THINKING_SUM = "<think>\n3 plus 4 is 7.\n</think>\n\n7"

CASES = [
    # name, capital reply, sum reply, G3 verdict, the CHAT / ARITHMETIC lines of the result file
    ("thinking, both answers after </think>", reply(THINKING_CAPITAL, 182), reply(THINKING_SUM, 155),
     "PASS", ["CHAT: Paris", "CHAT-META: tokens=182 finish=stop",
              "ARITHMETIC: 7", "ARITHMETIC-META: tokens=155 finish=stop"]),
    ("plain answers", reply("Paris"), reply("7"),
     "PASS", ["CHAT: Paris", "CHAT-META: tokens=10 finish=stop",
              "ARITHMETIC: 7", "ARITHMETIC-META: tokens=10 finish=stop"]),
    # the nightly L2 of 2026-11-25, Qwen3-32B, verbatim: the budget ended the reply inside the reasoning
    ("cut inside the reasoning (16 / 8 tokens)",
     reply("<think>\nOkay, the user is asking for the capital of France, and they", 16, "length"),
     reply("<think>\nOkay, let's see.", 8, "length"),
     "FAIL", ["CHAT: ", "CHAT-META: tokens=16 finish=length",
              "ARITHMETIC: ", "ARITHMETIC-META: tokens=8 finish=length"]),
    # the reasoning names Paris; the answer that follows is wrong
    ("Paris in the reasoning only", reply("<think>\nThe capital of France is Paris.\n</think>\n\nLondon"),
     reply(THINKING_SUM), "FAIL",
     ["CHAT: London", "CHAT-META: tokens=10 finish=stop",
      "ARITHMETIC: 7", "ARITHMETIC-META: tokens=10 finish=stop"]),
    ("wrong sum", reply("Paris"), reply("8"),
     "FAIL", ["CHAT: Paris", "CHAT-META: tokens=10 finish=stop",
              "ARITHMETIC: 8", "ARITHMETIC-META: tokens=10 finish=stop"]),
    ("17 is not 7", reply("Paris"), reply("17"),
     "FAIL", ["CHAT: Paris", "CHAT-META: tokens=10 finish=stop",
              "ARITHMETIC: 17", "ARITHMETIC-META: tokens=10 finish=stop"]),
    ("7 inside a sentence", reply("Paris"), reply("The answer is 7."),
     "PASS", ["CHAT: Paris", "CHAT-META: tokens=10 finish=stop",
              "ARITHMETIC: The answer is 7.", "ARITHMETIC-META: tokens=10 finish=stop"]),
    ("two think blocks, the last one counts",
     reply("<think>a</think>b</think>\nParis"), reply("<think>a</think>b</think>\n7"),
     "PASS", ["CHAT: Paris", "CHAT-META: tokens=10 finish=stop",
              "ARITHMETIC: 7", "ARITHMETIC-META: tokens=10 finish=stop"]),
    ("answer on several lines",
     reply("<think>\nEasy.\n</think>\n\nThe capital of France\nis Paris."), reply("<think>\n3 + 4\n= 7\n</think>\n\n7"),
     "PASS", ["CHAT: The capital of France is Paris.", "CHAT-META: tokens=10 finish=stop",
              "ARITHMETIC: 7", "ARITHMETIC-META: tokens=10 finish=stop"]),
    ("gateway error instead of a completion", "<html>502</html>", reply("7"),
     "FAIL", ["CHAT: ", "CHAT-META: parse-err: Expecting value: line 1 column 1 (char 0)",
              "ARITHMETIC: 7", "ARITHMETIC-META: tokens=10 finish=stop"]),
    ("empty answer", reply("Paris"), reply(""),
     "FAIL", ["CHAT: Paris", "CHAT-META: tokens=10 finish=stop",
              "ARITHMETIC: ", "ARITHMETIC-META: tokens=10 finish=stop"]),
]


def _judge(tmp_path, capital, sum_, g3_max_tokens=None):
    source = SUITE.read_text()
    start = source.index("gen_gate() {")
    end = source.index("\nITS=")
    # gen_gate, G3_MAX_TOKENS, chat_gate, answer_gate and judge_model as they are in the suite.
    # curl answers from two variables and logs every request body.
    script = 'set -u\n' + textwrap.dedent("""
        curl() {
          local url="" body=""
          while [ $# -gt 0 ]; do
            case "$1" in http://*) url=$1 ;; -d) shift; body=$1 ;; esac
            shift
          done
          printf '%s\\n' "$body" >> "$REQUESTS"
          case "$url" in
            */v1/chat/completions)
              case "$body" in *'capital of France'*) printf '%s' "$CAPITAL_REPLY" ;;
                                       *)            printf '%s' "$SUM_REPLY" ;;
              esac ;;
            *) printf '%s' "$GENERATE_REPLY" ;;
          esac
        }
        """) + source[start:end] + textwrap.dedent("""
        echo "q3-32b G1-launch PASS" > "$LOGD/result_q3-32b.txt"
        judge_model q3-32b /models/qwen3/Qwen3-32B
        """)
    requests = tmp_path / "requests.txt"
    generate = json.dumps({"text": "the capital of France. The Seine river divides the city into the "
                                   "Left Bank and the Right Bank. Paris"})
    env = dict(os.environ, LOGD=str(tmp_path), PORT="30036", REQUESTS=str(requests),
               CAPITAL_REPLY=capital, SUM_REPLY=sum_, GENERATE_REPLY=generate,
               HCU_ADAPTER_HOST="none", PYTHONPATH=str(ROOT),
               PATH=os.pathsep.join([str(Path(sys.executable).parent), os.environ["PATH"]]))
    if g3_max_tokens is not None:
        env["G3_MAX_TOKENS"] = g3_max_tokens
    run = subprocess.run(["bash", "-c", script], cwd=tmp_path, env=env,
                         capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stderr
    result = (tmp_path / "result_q3-32b.txt").read_text().splitlines()
    return result, [line for line in requests.read_text().splitlines() if line.startswith("{")]


@pytest.mark.parametrize("name,capital,sum_,verdict,g3_lines", CASES, ids=[c[0] for c in CASES])
def test_g3_verdict(tmp_path, name, capital, sum_, verdict, g3_lines):
    result, _ = _judge(tmp_path, capital, sum_)
    # the caller writes the G1 line before judge_model runs: it has to survive the judgement
    assert result[0] == "q3-32b G1-launch PASS"
    assert result[2] == "G2-generation PASS"
    assert result[3:7] == g3_lines
    assert result[7] == "G3-precision " + verdict
    assert result[-1] == "G4-long PASS"


def _chat_requests(requests):
    return [json.loads(line) for line in requests if '"messages"' in line]


def test_chat_requests_carry_the_budget_and_the_anchors(tmp_path):
    _, requests = _judge(tmp_path, reply(THINKING_CAPITAL), reply(THINKING_SUM))
    chats = _chat_requests(requests)
    # a thinking model needs hundreds of tokens before its answer; 16 cut every Qwen3 reply
    assert [c["max_tokens"] for c in chats] == [2048, 2048]
    assert [c["messages"][0]["content"] for c in chats] == [CAPITAL, SUM]
    assert all(c["temperature"] == 0 for c in chats)


def test_chat_budget_can_be_overridden(tmp_path):
    _, requests = _judge(tmp_path, reply("Paris"), reply("7"), g3_max_tokens="64")
    assert [c["max_tokens"] for c in _chat_requests(requests)] == [64, 64]
