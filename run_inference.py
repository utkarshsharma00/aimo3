#!/usr/bin/env python3
"""
AIMO3 Standalone Inference — flamingice (solo)
==============================================
Kaggle-agnostic version of final_submission v2.
Runs on any machine with an H100 80GB GPU.

Model: danielhanchen/gpt-oss-120b (unmodified, no fine-tuning)
Method: 8 parallel tool-augmented attempts with 5-component weighted
        entropy voting and adaptive time budgeting.

Usage:
    # Install dependencies
    pip install vllm openai openai_harmony transformers jupyter_client polars

    # Download model (one-time, ~60GB)
    huggingface-cli download danielhanchen/gpt-oss-120b --local-dir ./gpt-oss-120b

    # Run on a CSV of problems (columns: id, problem)
    python run_inference.py --model_path ./gpt-oss-120b --input problems.csv --output answers.csv

    # Or run on a single problem
    python run_inference.py --model_path ./gpt-oss-120b --problem "Find the remainder when 2^100 is divided by 7."

Requirements:
    - 1x H100 80GB (or equivalent with >=80GB VRAM)
    - ~60GB disk for model weights
    - Python 3.10+
"""

import os
import sys
import gc
import re
import math
import time
import queue
import threading
import subprocess
import contextlib
import argparse
import csv
from typing import Optional
from collections import Counter, defaultdict
from concurrent.futures import as_completed, ThreadPoolExecutor

from jupyter_client import KernelManager
from transformers import set_seed


# ============================================================
# Configuration
# ============================================================

class CFG:
    system_prompt = (
        'You are a world-class IMO competitor. '
        'The final answer must be a non-negative integer between 0 and 99999. '
        'Place your final answer inside \\boxed{}.'
    )
    tool_prompt = (
        'Use this tool to execute Python code. '
        'The environment is a stateful Jupyter notebook. '
        'Always use print() to output results.'
    )
    preference_prompt = (
        'You have access to math, numpy, sympy, and mpmath.'
    )

    served_model_name = 'gpt-oss'
    model_path = None  # Set via CLI

    temperature = 1.0
    min_p = 0.02

    kv_cache_dtype = 'fp8_e4m3'
    dtype = 'auto'
    context_tokens = 65536
    gpu_memory_utilization = 0.96
    batch_size = 256
    stream_interval = 200
    buffer_tokens = 512
    search_tokens = 32
    top_logprobs = 5

    attempts = 8
    early_stop = 4
    workers = 16
    turns = 128
    seed = 42

    high_problem_timeout = 900
    base_problem_timeout = 270
    notebook_limit = 17400
    server_timeout = 180
    session_timeout = 960
    jupyter_timeout = 6
    sandbox_timeout = 3


# ============================================================
# Template (Harmony encoding)
# ============================================================

from openai import OpenAI
from openai_harmony import (
    HarmonyEncodingName, load_harmony_encoding, SystemContent,
    ReasoningEffort, ToolNamespaceConfig, Author, Message,
    Role, TextContent, Conversation
)


class AIMO3Template:
    def get_system_content(self, system_prompt, tool_config):
        return (
            SystemContent.new()
            .with_model_identity(system_prompt)
            .with_reasoning_effort(reasoning_effort=ReasoningEffort.HIGH)
            .with_tools(tool_config)
        )

    def apply_chat_template(self, system_prompt, user_prompt, tool_config):
        system_content = self.get_system_content(system_prompt, tool_config)
        system_message = Message.from_role_and_content(Role.SYSTEM, system_content)
        user_message = Message.from_role_and_content(Role.USER, user_prompt)
        return [system_message, user_message]


# ============================================================
# Sandbox (Jupyter kernel)
# ============================================================

class AIMO3Sandbox:
    _port_lock = threading.Lock()
    _next_port = 50000

    @classmethod
    def _get_next_ports(cls, count=5):
        with cls._port_lock:
            ports = list(range(cls._next_port, cls._next_port + count))
            cls._next_port += count
            return ports

    def __init__(self, timeout):
        self._default_timeout = timeout
        self._owns_kernel = False
        self._client = None
        self._km = None

        ports = self._get_next_ports(5)

        env = os.environ.copy()
        env['PYDEVD_DISABLE_FILE_VALIDATION'] = '1'
        env['PYDEVD_WARN_EVALUATION_TIMEOUT'] = '0'
        env['JUPYTER_PLATFORM_DIRS'] = '1'
        env['PYTHONWARNINGS'] = 'ignore'
        env['MPLBACKEND'] = 'Agg'

        self._km = KernelManager()
        self._km.shell_port = ports[0]
        self._km.iopub_port = ports[1]
        self._km.stdin_port = ports[2]
        self._km.hb_port = ports[3]
        self._km.control_port = ports[4]

        self._km.start_kernel(env=env, extra_arguments=['--Application.log_level=CRITICAL'])
        self._client = self._km.blocking_client()
        self._client.start_channels()
        self._client.wait_for_ready(timeout=self._default_timeout)
        self._owns_kernel = True

        self.execute(
            'import math\n'
            'import numpy\n'
            'import sympy\n'
            'import mpmath\n'
            'import itertools\n'
            'import collections\n'
            'mpmath.mp.dps = 64\n'
        )

    def _format_error(self, traceback):
        clean_lines = []
        for frame in traceback:
            clean_frame = re.sub(r'\x1b\[[0-9;]*m', '', frame)
            if 'File "' in clean_frame and 'ipython-input' not in clean_frame:
                continue
            clean_lines.append(clean_frame)
        return ''.join(clean_lines)

    def execute(self, code, timeout=None):
        client = self._client
        effective_timeout = timeout or self._default_timeout
        msg_id = client.execute(code, store_history=True, allow_stdin=False, stop_on_error=False)

        stdout_parts, stderr_parts = [], []
        start_time = time.time()

        while True:
            if time.time() - start_time > effective_timeout:
                self._km.interrupt_kernel()
                return f'[ERROR] Execution timed out after {effective_timeout} seconds'
            try:
                msg = client.get_iopub_msg(timeout=1.0)
            except queue.Empty:
                continue

            if msg.get('parent_header', {}).get('msg_id') != msg_id:
                continue

            msg_type = msg.get('msg_type')
            content = msg.get('content', {})

            if msg_type == 'stream':
                text = content.get('text', '')
                if content.get('name') == 'stdout':
                    stdout_parts.append(text)
                else:
                    stderr_parts.append(text)
            elif msg_type == 'error':
                stderr_parts.append(self._format_error(content.get('traceback', [])))
            elif msg_type in {'execute_result', 'display_data'}:
                data = content.get('data', {})
                text = data.get('text/plain')
                if text:
                    stdout_parts.append(text if text.endswith('\n') else f'{text}\n')
            elif msg_type == 'status':
                if content.get('execution_state') == 'idle':
                    break

        stdout = ''.join(stdout_parts)
        stderr = ''.join(stderr_parts)
        if stderr:
            return f'{stdout.rstrip()}\n{stderr}' if stdout else stderr
        return stdout if stdout.strip() else '[WARN] No output. Use print() to see results.'

    def close(self):
        with contextlib.suppress(Exception):
            if self._client:
                self._client.stop_channels()
        if self._owns_kernel and self._km is not None:
            with contextlib.suppress(Exception):
                self._km.shutdown_kernel(now=True)
            with contextlib.suppress(Exception):
                self._km.cleanup_resources()

    def reset(self):
        self.execute(
            '%reset -f\n'
            'import math\nimport numpy\nimport sympy\nimport mpmath\n'
            'import itertools\nimport collections\nmpmath.mp.dps = 64\n'
        )

    def __del__(self):
        self.close()


# ============================================================
# Tool (Python execution)
# ============================================================

class AIMO3Tool:
    def __init__(self, local_jupyter_timeout, tool_prompt, sandbox=None):
        self._local_jupyter_timeout = local_jupyter_timeout
        self._tool_prompt = tool_prompt
        self._jupyter_session = sandbox
        self._owns_session = sandbox is None
        self._execution_lock = threading.Lock()
        self._init_lock = threading.Lock()

    def _ensure_session(self):
        if self._jupyter_session is None:
            with self._init_lock:
                if self._jupyter_session is None:
                    self._jupyter_session = AIMO3Sandbox(timeout=self._local_jupyter_timeout)

    def _ensure_last_print(self, code):
        lines = code.strip().split('\n')
        if not lines:
            return code
        last_line = lines[-1].strip()
        if 'print' in last_line or 'import' in last_line:
            return code
        if not last_line or last_line.startswith('#'):
            return code
        lines[-1] = 'print(' + last_line + ')'
        return '\n'.join(lines)

    @property
    def instruction(self):
        return self._tool_prompt

    @property
    def tool_config(self):
        return ToolNamespaceConfig(name='python', description=self.instruction, tools=[])

    def _make_response(self, output, channel=None):
        content = TextContent(text=output)
        author = Author(role=Role.TOOL, name='python')
        message = Message(author=author, content=[content]).with_recipient('assistant')
        if channel:
            message = message.with_channel(channel)
        return message

    def process_sync_plus(self, message):
        self._ensure_session()
        raw_script = message.content[0].text
        final_script = self._ensure_last_print(raw_script)
        with self._execution_lock:
            try:
                output = self._jupyter_session.execute(final_script)
            except TimeoutError as exc:
                output = f'[ERROR] {exc}'
        return [self._make_response(output, channel=message.channel)]


# ============================================================
# Solver (main inference engine)
# ============================================================

class AIMO3Solver:

    def __init__(self, cfg, port=8000):
        self.cfg = cfg
        self.port = port
        self.base_url = f'http://0.0.0.0:{port}/v1'
        self.api_key = 'sk-local'
        self.template = AIMO3Template()
        self.encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        self.stop_token_ids = self.encoding.stop_tokens_for_assistant_actions()

        self._preload_model_weights()
        self.server_process = self._start_server()
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=self.cfg.session_timeout)
        self._wait_for_server()
        self._initialize_kernels()

        self.notebook_start_time = time.time()
        self.problems_remaining = 50
        self.problem_counter = 0

    def _preload_model_weights(self):
        print(f'Loading model weights from {self.cfg.model_path}...')
        t0 = time.time()
        files_to_load, total_size = [], 0
        for root, _, files in os.walk(self.cfg.model_path):
            for fn in files:
                fp = os.path.join(root, fn)
                if os.path.isfile(fp):
                    files_to_load.append(fp)
                    total_size += os.path.getsize(fp)
        def _read(path):
            with open(path, 'rb') as f:
                while f.read(1024*1024*1024): pass
        with ThreadPoolExecutor(max_workers=self.cfg.workers) as ex:
            list(ex.map(_read, files_to_load))
        print(f'Loaded {len(files_to_load)} files ({total_size/1e9:.1f}GB) in {time.time()-t0:.1f}s')

    def _start_server(self):
        cmd = [
            sys.executable, '-m', 'vllm.entrypoints.openai.api_server',
            '--seed', str(self.cfg.seed),
            '--model', self.cfg.model_path,
            '--served-model-name', self.cfg.served_model_name,
            '--tensor-parallel-size', '1',
            '--max-num-seqs', str(self.cfg.batch_size),
            '--gpu-memory-utilization', str(self.cfg.gpu_memory_utilization),
            '--host', '0.0.0.0', '--port', str(self.port),
            '--dtype', self.cfg.dtype,
            '--kv-cache-dtype', self.cfg.kv_cache_dtype,
            '--max-model-len', str(self.cfg.context_tokens),
            '--stream-interval', str(self.cfg.stream_interval),
            '--async-scheduling', '--disable-log-stats',
            '--enable-prefix-caching',
        ]
        self.log_file = open('vllm_server.log', 'w')
        return subprocess.Popen(cmd, stdout=self.log_file, stderr=subprocess.STDOUT, start_new_session=True)

    def _wait_for_server(self):
        print('Waiting for vLLM server...')
        t0 = time.time()
        for _ in range(self.cfg.server_timeout):
            if self.server_process.poll() is not None:
                self.log_file.flush()
                with open('vllm_server.log') as f:
                    raise RuntimeError(f'Server died. Logs:\n{f.read()}')
            try:
                self.client.models.list()
                print(f'Server ready in {time.time()-t0:.1f}s')
                return
            except Exception:
                time.sleep(1)
        raise RuntimeError('Server startup timeout')

    def _initialize_kernels(self):
        print(f'Initializing {self.cfg.workers} Jupyter kernels...')
        t0 = time.time()
        self.sandbox_pool = queue.Queue()
        created = 0
        for i in range(self.cfg.workers):
            for attempt in range(3):
                try:
                    sb = AIMO3Sandbox(timeout=self.cfg.jupyter_timeout)
                    self.sandbox_pool.put(sb)
                    created += 1
                    break
                except Exception as e:
                    if attempt < 2:
                        time.sleep(1)
                    else:
                        print(f'  Kernel {i} failed after 3 attempts: {e}')
        print(f'Kernels ready: {created}/{self.cfg.workers} in {time.time()-t0:.1f}s')

    def _scan_for_answer(self, text):
        pattern = r'\\boxed\s*\{\s*([0-9,]+)\s*\}'
        matches = re.findall(pattern, text)
        if matches:
            try:
                val = int(matches[-1].replace(',', ''))
                if 0 <= val <= 99999:
                    return val
            except ValueError:
                pass
        pattern = r'final\s+answer\s+is\s*([0-9,]+)'
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            try:
                val = int(matches[-1].replace(',', ''))
                if 0 <= val <= 99999:
                    return val
            except ValueError:
                pass
        return None

    def _compute_weighted_entropy(self, logprobs_buffer):
        """5-component weighted entropy for vote quality estimation."""
        if not logprobs_buffer:
            return float('inf')
        entropies = []
        for top_lp in logprobs_buffer:
            if not isinstance(top_lp, dict) or not top_lp:
                continue
            h = 0.0
            for _, lp in top_lp.items():
                p = math.exp(lp)
                if p > 0:
                    h -= p * math.log2(p)
            entropies.append(h)
        if not entropies:
            return float('inf')
        n = len(entropies)
        mean_ent = sum(entropies) / n
        if n > 1:
            variance = sum((e - mean_ent) ** 2 for e in entropies) / (n - 1)
            std_dev = math.sqrt(variance)
        else:
            std_dev = 0.0
        decay = 0.995
        weights = [decay ** (n - 1 - i) for i in range(n)]
        total_weight = sum(weights)
        position_weighted = sum(w * e for w, e in zip(weights, entropies)) / total_weight if total_weight > 0 else mean_ent
        high_ent_count = sum(1 for e in entropies if e > 2.0)
        high_ent_ratio = high_ent_count / n
        max_streak, current_streak = 0, 0
        for e in entropies:
            if e < 1.0:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        streak_bonus = -0.1 * (max_streak / n) if n > 0 else 0.0
        weighted = (
            0.3 * mean_ent +
            0.4 * position_weighted +
            0.2 * std_dev +
            0.3 * high_ent_ratio * 3.0 +
            streak_bonus
        )
        return max(weighted, 1e-9)

    def _process_attempt(self, problem, system_prompt, attempt_index, stop_event, deadline):
        if stop_event.is_set() or time.time() > deadline:
            return {'Attempt': attempt_index + 1, 'Answer': None, 'Python Calls': 0,
                    'Python Errors': 0, 'Response Length': 0, 'Entropy': float('inf')}

        local_tool = None
        sandbox = None
        python_calls = 0
        python_errors = 0
        total_tokens = 0
        final_answer = None
        logprobs_buffer = []
        attempt_seed = int(math.pow(self.cfg.seed + attempt_index, 2))

        try:
            sandbox = self.sandbox_pool.get(timeout=self.cfg.sandbox_timeout)
            local_tool = AIMO3Tool(
                local_jupyter_timeout=self.cfg.jupyter_timeout,
                tool_prompt=self.cfg.tool_prompt,
                sandbox=sandbox
            )
            messages = self.template.apply_chat_template(
                system_prompt, problem, local_tool.tool_config
            )
            conversation = Conversation.from_messages(messages)

            for _ in range(self.cfg.turns):
                if stop_event.is_set() or time.time() > deadline:
                    break
                prompt_ids = self.encoding.render_conversation_for_completion(conversation, Role.ASSISTANT)
                max_tokens = self.cfg.context_tokens - len(prompt_ids)
                if max_tokens < self.cfg.buffer_tokens:
                    break

                stream = self.client.completions.create(
                    model=self.cfg.served_model_name,
                    temperature=self.cfg.temperature,
                    logprobs=self.cfg.top_logprobs,
                    max_tokens=max_tokens,
                    prompt=prompt_ids,
                    seed=attempt_seed,
                    stream=True,
                    extra_body={
                        'min_p': self.cfg.min_p,
                        'stop_token_ids': self.stop_token_ids,
                        'return_token_ids': True
                    }
                )

                try:
                    token_buffer = []
                    text_chunks = []
                    for chunk in stream:
                        if stop_event.is_set() or time.time() > deadline:
                            break
                        new_tokens = chunk.choices[0].token_ids
                        new_text = chunk.choices[0].text
                        if new_tokens:
                            token_buffer.extend(new_tokens)
                            total_tokens += len(new_tokens)
                            text_chunks.append(new_text)
                            chunk_lp = chunk.choices[0].logprobs
                            if chunk_lp and chunk_lp.top_logprobs:
                                logprobs_buffer.extend(chunk_lp.top_logprobs)
                        if new_text and '}' in new_text:
                            search_text = ''.join(text_chunks[-self.cfg.search_tokens:])
                            answer = self._scan_for_answer(search_text)
                            if answer is not None:
                                final_answer = answer
                                break
                finally:
                    stream.close()

                if final_answer is not None:
                    break
                if not token_buffer:
                    break

                new_messages = self.encoding.parse_messages_from_completion_tokens(token_buffer, Role.ASSISTANT)
                conversation.messages.extend(new_messages)
                if not new_messages:
                    break
                last_message = new_messages[-1]
                if last_message.channel == 'final':
                    final_answer = self._scan_for_answer(last_message.content[0].text)
                    break
                if last_message.recipient == 'python':
                    python_calls += 1
                    tool_responses = local_tool.process_sync_plus(last_message)
                    response_text = tool_responses[0].content[0].text
                    if response_text.startswith('[ERROR]') or 'Traceback' in response_text or 'Error:' in response_text:
                        python_errors += 1
                    conversation.messages.extend(tool_responses)

        except Exception:
            python_errors += 1
        finally:
            if sandbox is not None:
                sandbox.reset()
                self.sandbox_pool.put(sandbox)

        return {
            'Attempt': attempt_index + 1,
            'Response Length': total_tokens,
            'Python Calls': python_calls,
            'Python Errors': python_errors,
            'Entropy': self._compute_weighted_entropy(logprobs_buffer),
            'Answer': final_answer,
        }

    def _select_answer(self, detailed_results):
        """Inverse entropy weighted voting."""
        answer_weights = defaultdict(float)
        answer_votes = defaultdict(int)
        for r in detailed_results:
            a = r['Answer']
            e = r['Entropy']
            if a is not None:
                answer_weights[a] += 1.0 / max(e, 1e-9)
                answer_votes[a] += 1
        if not answer_weights:
            return 0
        scored = sorted(
            [{'answer': a, 'votes': answer_votes[a], 'score': w}
             for a, w in answer_weights.items()],
            key=lambda x: x['score'], reverse=True
        )
        print(f'  Votes: {[(s["answer"], s["votes"], round(s["score"], 2)) for s in scored[:5]]}')
        return scored[0]['answer']

    def solve_problem(self, problem):
        self.problem_counter += 1
        pnum = self.problem_counter
        print(f'\n{"="*60}')
        print(f'Problem {pnum}')
        print(f'{"="*60}')

        user_input = f'{problem} {self.cfg.preference_prompt}'

        elapsed = time.time() - self.notebook_start_time
        time_left = self.cfg.notebook_limit - elapsed
        reserved = max(0, self.problems_remaining - 1) * self.cfg.base_problem_timeout
        budget = min(max(time_left - reserved, self.cfg.base_problem_timeout), self.cfg.high_problem_timeout)
        deadline = time.time() + budget
        print(f'[BUDGET] {budget:.0f}s | Remaining: {self.problems_remaining}')

        detailed_results = []
        valid_answers = []
        stop_event = threading.Event()
        early_stopped = False

        executor = ThreadPoolExecutor(max_workers=self.cfg.workers)
        try:
            futures = [
                executor.submit(
                    self._process_attempt, user_input,
                    self.cfg.system_prompt,
                    i, stop_event, deadline
                )
                for i in range(self.cfg.attempts)
            ]
            for future in as_completed(futures):
                try:
                    result = future.result()
                    detailed_results.append(result)
                    if result['Answer'] is not None:
                        valid_answers.append(result['Answer'])
                    counts = Counter(valid_answers).most_common(1)
                    if counts and counts[0][1] >= self.cfg.early_stop:
                        early_stopped = True
                        stop_event.set()
                        for f in futures:
                            f.cancel()
                        break
                except Exception as e:
                    print(f'  Attempt err: {e}')
        finally:
            stop_event.set()
            executor.shutdown(wait=True, cancel_futures=True)

        self.problems_remaining = max(0, self.problems_remaining - 1)

        if not valid_answers:
            print(f'[P{pnum:02d}] No valid answers -> 0')
            return 0

        final_answer = self._select_answer(detailed_results)
        print(f'[P{pnum:02d}] answer={final_answer} ({len(detailed_results)} attempts, early={early_stopped})')
        return int(final_answer)

    def __del__(self):
        if hasattr(self, 'server_process'):
            self.server_process.terminate()
            self.server_process.wait()
        if hasattr(self, 'log_file'):
            self.log_file.close()
        if hasattr(self, 'sandbox_pool'):
            while not self.sandbox_pool.empty():
                try:
                    self.sandbox_pool.get_nowait().close()
                except Exception:
                    pass


# ============================================================
# CLI entry point
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='AIMO3 Inference — flamingice')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to gpt-oss-120b model directory')
    parser.add_argument('--input', type=str, default=None,
                        help='CSV file with columns: id, problem')
    parser.add_argument('--output', type=str, default='answers.csv',
                        help='Output CSV file (default: answers.csv)')
    parser.add_argument('--problem', type=str, default=None,
                        help='Single problem text (alternative to --input)')
    parser.add_argument('--port', type=int, default=8000,
                        help='vLLM server port (default: 8000)')
    parser.add_argument('--num_problems', type=int, default=50,
                        help='Total expected problems for time budgeting (default: 50)')
    args = parser.parse_args()

    os.environ['TRANSFORMERS_NO_TF'] = '1'
    os.environ['TRANSFORMERS_NO_FLAX'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    CFG.model_path = args.model_path
    set_seed(CFG.seed)

    solver = AIMO3Solver(CFG, port=args.port)
    solver.problems_remaining = args.num_problems

    if args.problem:
        # Single problem mode
        answer = solver.solve_problem(args.problem)
        print(f'\n{"="*60}')
        print(f'FINAL ANSWER: {answer}')
        print(f'{"="*60}')

    elif args.input:
        # CSV mode
        problems = []
        with open(args.input, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                problems.append((row['id'], row['problem']))

        solver.problems_remaining = len(problems)
        results = []

        for pid, problem_text in problems:
            gc.disable()
            try:
                answer = solver.solve_problem(problem_text)
            except Exception as e:
                print(f'Error on {pid}: {e}')
                answer = 0
            gc.enable()
            gc.collect()
            results.append({'id': pid, 'answer': answer})

        with open(args.output, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['id', 'answer'])
            writer.writeheader()
            writer.writerows(results)

        print(f'\nResults written to {args.output}')
        print(f'Solved {len(results)} problems')
    else:
        print('Error: provide either --input (CSV) or --problem (single problem)')
        sys.exit(1)


if __name__ == '__main__':
    main()
