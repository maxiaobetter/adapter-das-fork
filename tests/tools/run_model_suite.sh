#!/bin/bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# Model-suite batch runner: wait for a free HCU, then validate each model in
# docs/models.md, one at a time, four-gate each (launch/generation/precision/
# long-context). State + logs under /root/logs/model_suite.
# Usage (inside pr418_test):  bash /root/run_model_suite.sh [model-id ...]
set -u
export LD_LIBRARY_PATH=/opt/dtk/.hyhal/rocm_smi/lib:$LD_LIBRARY_PATH
PORT=30036
LOGD=/root/logs/model_suite
mkdir -p $LOGD
SGLANG_V0521="${SGLANG_V0521:-$HOME/sglang-v0521/python}"
ADAPTER="${ADAPTER:-/root/hcu_t2}"
export PYTHONPATH=$ADAPTER:$SGLANG_V0521
export SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=python
export SGLANG_RUST_BUILD_MODE=never
export HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0
export TVM_FFI_DISABLE_TORCH_C_DLPACK=1

M=/public/opendas/DL_DATA/llm-models

declare -A MODELS=(
  # id | model_path | extra args
  [q25-05b]="$M/qwen2.5/Qwen2.5-0.5B-Instruct::"
  [q3-32b]="$M/qwen3/Qwen3-32B::--mem-fraction-static 0.85"
  [q3-30b-a3b]="$M/qwen3/Qwen3-30B-A3B::--mem-fraction-static 0.85"
  [q38-27b-fp8]="$M/qwen3.8/Qwen3.8-27B-FP8::--mem-fraction-static 0.85 --trust-remote-code"
  [q35-122b-awq]="$M/qwen3.5/Qwen3.5-122B-A10B-AWQ::--tp-size 4 --mem-fraction-static 0.85 --trust-remote-code"
)
ORDER=(q25-05b q3-32b q3-30b-a3b q38-27b-fp8 q35-122b-awq)

need_mib() {  # rough VRAM need per model id (MiB), includes graph/act margin
  case $1 in
    q25-05b) echo 4200 ;;
    q3-32b) echo 78000 ;;
    q3-30b-a3b) echo 74000 ;;
    q38-27b-fp8) echo 40000 ;;
    q35-122b-awq) echo 50000 ;;   # per-GPU with TP4
    *) echo 999999 ;;
  esac
}

free_mib() {  # free VRAM (MiB) of HCU[$1]
  # unit-adaptive: DTK images differ -- older ones report "(KiB)", the
  # 0.5.22-dtk2610 image reports "(MiB)". Dividing a MiB reading by 1024
  # again made every idle card read as 64 MiB free and the suite waited
  # forever on an empty machine (2026-11-19 bw47, 7/8 cards idle).
  local out tline uline t u
  out=$(/opt/dtk/bin/rocm-smi --showmeminfo vram 2>/dev/null)
  tline=$(echo "$out" | grep -A1 "HCU\[$1\]" | grep "Total Memory" | head -1 | tr -d ' ')
  uline=$(echo "$out" | grep -A2 "HCU\[$1\]" | grep "Used Memory" | head -1 | tr -d ' ')
  t=${tline##*:}; u=${uline##*:}
  case "$tline" in
    *"(KiB)"*) echo $(( (t - u) / 1024 )) ;;
    *)         echo $(( t - u )) ;;
  esac
}

stop_server() {
  pkill -9 -f "sglang.launch_server.*--port $PORT" 2>/dev/null
  pkill -9 -f "sglang::" 2>/dev/null
  sleep 6
}

start_server() {  # $1 model-path, $2 extra args
  stop_server
  echo "[suite] launching: $1 $2"
  nohup python -m sglang.launch_server \
    --model-path "$1" --host 0.0.0.0 --port $PORT \
    --page-size 64 --log-level info $2 \
    > $LOGD/launch_${1##*/}.log 2>&1 &
}

wait_ready() {  # $1 timeout seconds
  # ready = HTTP 2xx (curl -f), any body: this sglang's /health answers
  # 200 with a NON-empty body, and the old `grep -q '^$'` (empty-body only)
  # never matched -- the suite idled the full window and then killed the
  # perfectly healthy server (bw47 2026-11-19, G1 false FAIL).
  local t=0
  while [ $t -lt $1 ]; do
    if curl -sf --max-time 3 http://127.0.0.1:$PORT/health >/dev/null 2>&1; then
      sleep 2
      curl -sf --max-time 3 http://127.0.0.1:$PORT/health >/dev/null 2>&1 && return 0
    fi
    # dead early?
    if ! pgrep -f "sglang.launch_server.*--port $PORT" >/dev/null; then
      return 2
    fi
    sleep 10; t=$((t + 10))
  done
  return 1
}

gen_gate() {  # $2 prompt -> prints generated text tail
  curl -s --max-time 90 http://127.0.0.1:$PORT/generate -H 'Content-Type: application/json' \
    -d "{\"text\": \"$2\", \"sampling_params\": {\"max_new_tokens\": 32, \"temperature\": 0}}" \
  | python3 -c 'import json,sys
try:
    print(json.load(sys.stdin).get("text", "")[-160:])
except Exception as e:
    print("parse-err:", e)'
}

G3_MAX_TOKENS="${G3_MAX_TOKENS:-2048}"   # room for a thinking model to finish </think> and answer

chat_gate() {  # $1 user message -> chat completion JSON, greedy
  local msg=${1:?user message}
  curl -s --max-time 900 http://127.0.0.1:$PORT/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d "{\"model\": \"x\", \"messages\": [{\"role\": \"user\", \"content\": \"$msg\"}], \"max_tokens\": $G3_MAX_TOKENS, \"temperature\": 0}"
}

answer_gate() {  # stdin: chat completion JSON -> the judged text on one line, then one "META ..." line
  # The judged text is the answer, i.e. the text after the last </think>. No
  # <think> at all: the whole content. <think> opened but not closed (the
  # reply was cut inside the reasoning): nothing -- a gate must not be passed
  # on reasoning text. The META line says how many tokens were generated and
  # how the reply ended, so that a cut or failed reply can be told from a
  # wrong one when reading the result file.
  python3 -c 'import json,sys
try:
    d = json.load(sys.stdin)
    choice = d["choices"][0]
    text = choice["message"]["content"] or ""
    meta = "META tokens=%s finish=%s" % (d.get("usage", {}).get("completion_tokens"), choice.get("finish_reason"))
except Exception as e:
    sys.stderr.write("parse-err: %s\n" % e); text, meta = "", "META parse-err: %s" % e
if "<think>" in text and "</think>" not in text:
    sys.stderr.write("cut inside <think> (%d chars)\n" % len(text)); text = ""
else:
    text = text.rsplit("</think>", 1)[-1]
print(" ".join(text.split())); print(meta)'
}

judge_model() {  # $1 id, $2 model-path
  local id=$1 name=${2##*/}
  local res=$LOGD/result_$id.txt

  # G1 launch handled by caller; G2 generation
  local short short_out
  short_out=$(gen_gate short "The distance from London to Paris is 344 km. Distance from Paris to Madrid is")
  echo "GEN: $short_out" >> $res
  if [ "$(echo "$short_out" | tr -d '[:space:]' | wc -c)" -gt 3 ]; then
    echo "G2-generation PASS" >> $res
  else
    echo "G2-generation FAIL" >> $res
  fi

  # G3 precision: two independent semantic anchors via /v1/chat/completions.
  # A single geography answer can pass while a broken decode path stops on
  # arithmetic, so require both a knowledge and an arithmetic answer.
  # Thinking models (Qwen3 etc.) open with <think>...</think>: the budget
  # must let them finish thinking, and only the text after the closing tag
  # is judged -- the reasoning says "Paris" whether or not an answer
  # follows (docs/cookbook.md section 4, item 4). With 16 / 8 tokens every
  # Qwen3 reply was cut inside the reasoning (nightly L2, 2026-11-25).
  # The arithmetic anchor is a plain sum, not the word problem used before
  # ("a box has 12 apples, five are removed"): Qwen2.5-0.5B-Instruct
  # answers that with "12" and stops, whatever the budget (adapter-free HF
  # reference, first token "1" at 0.38 against "7" at 0.14; bw47
  # 2026-11-25), so it measured the model rather than the serving path.
  # answer_gate prints the judged text, then one "META ..." line. The judged
  # text is on one line, so it is everything but the last line of the output.
  local chat chat_out chat_text chat_meta arithmetic arithmetic_out arithmetic_text arithmetic_meta
  chat=$(chat_gate "What is the capital of France? Answer in one word.")
  chat_out=$(echo "$chat" | answer_gate)
  chat_text=${chat_out%$'\n'*}; chat_meta=${chat_out##*$'\n'}
  echo "CHAT: $(printf '%s\n' "$chat_text" | tail -c 240)" >> $res
  echo "CHAT-META: ${chat_meta#META }" >> $res
  arithmetic=$(chat_gate "What is 3 plus 4? Answer with only the number.")
  arithmetic_out=$(echo "$arithmetic" | answer_gate)
  arithmetic_text=${arithmetic_out%$'\n'*}; arithmetic_meta=${arithmetic_out##*$'\n'}
  echo "ARITHMETIC: $(printf '%s\n' "$arithmetic_text" | tail -c 240)" >> $res
  echo "ARITHMETIC-META: ${arithmetic_meta#META }" >> $res
  if echo "$chat_text" | grep -qi paris && \
     echo "$arithmetic_text" | grep -qE '(^|[^0-9])7([^0-9]|$)'; then
    echo "G3-precision PASS" >> $res
  else
    echo "G3-precision FAIL" >> $res
  fi

  # G4 long context ~2700 tokens (sparse variants only; dense fine too)
  local long_out
  long_out=$(curl -s --max-time 240 http://127.0.0.1:$PORT/generate \
    -H 'Content-Type: application/json' \
    -d @- <<REQEOF
{"text": "$(printf 'Paris is the capital of France. The Seine river divides the city into the Left Bank and the Right Bank. %.0s' $(seq 1 260))", "sampling_params": {"max_new_tokens": 24, "temperature": 0}}
REQEOF
)
  echo "LONG: $(echo "$long_out" | python3 -c 'import json,sys
try:
    print(json.load(sys.stdin).get("text","")[-100:].replace(chr(10)," "))
except Exception as e:
    print("parse-err", e)' 2>/dev/null)" >> $res
  if echo "$long_out" | python3 -c 'import json,sys
try:
    t = json.load(sys.stdin).get("text","")
    sys.exit(0 if len(t.strip()) > 8 else 1)
except Exception:
    sys.exit(1)' 2>/dev/null; then
    echo "G4-long PASS" >> $res
  else
    echo "G4-long FAIL" >> $res
  fi
}

ITS="${*:-${ORDER[@]}}"
echo "[suite] models: $ITS"

for id in $ITS; do
  spec=${MODELS[$id]}
  path=${spec%%::*}
  extra=${spec##*::}
  need=$(need_mib $id)

  echo "[suite] [$id] waiting for >= ${need}MiB free on any HCU..."
  while true; do
    avail=0
    for card in 0 1 2 3 4 5 6 7; do
      a=$(free_mib $card)
      if [ "${a:-0}" -ge "$need" ]; then avail=$a
        export HIP_VISIBLE_DEVICES=$card CUDA_VISIBLE_DEVICES=$card
        break
      fi
    done
    if [ "$avail" -ge "$need" ]; then break; fi
    sleep 120
  done

  echo "[suite] [$id] starting (avail=${avail}MiB)"
  # One fresh result file per model: judge_model appends, so the G1 line
  # written below has to start from an empty file.
  : > $LOGD/result_$id.txt
  start_server "$path" "$extra"
  wait_ready 1500
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "$id G1-launch FAIL (wait rc=$rc)" >> $LOGD/result_$id.txt
    # The end of the launch log travels with the result: the Verdict step
    # prints result files, so the cause shows up in the job log itself.
    echo "launch log tail:" >> $LOGD/result_$id.txt
    tail -n 12 $LOGD/launch_${path##*/}.log >> $LOGD/result_$id.txt 2>/dev/null
    echo "[suite] [$id] LAUNCH FAILED, next"
    continue
  fi
  echo "$id G1-launch PASS" >> $LOGD/result_$id.txt
  judge_model "$id" "$path"
  echo "[suite] [$id] verified: $(grep -c PASS $LOGD/result_$id.txt)/4 gates"
  tail -5 $LOGD/result_$id.txt
done
stop_server
echo "[suite] DONE"
