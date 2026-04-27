#!/bin/bash
# Side-by-side benchmark: transformers reference vs nano-cohere-transcribe.
# Usage: ./benchmark.sh <audio_file> [language]
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

AUDIO="${1:-}"
LANG="${2:-en}"

if [ -z "$AUDIO" ]; then
    echo "Usage: $0 <audio_file> [language]"
    echo "  language default: en. Supported: en fr de es it pt nl pl el ar ja zh vi ko"
    exit 1
fi

PY="${PYTHON:-python}"

"$PY" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null || {
    echo "ERROR: PyTorch with CUDA required."
    exit 1
}

GPU=$("$PY" -c 'import torch; print(torch.cuda.get_device_name(0))')
echo "=== Cohere Transcribe benchmark ==="
echo "GPU:      $GPU"
echo "PyTorch:  $("$PY" -c 'import torch; print(torch.__version__)')"
echo "CUDA:     $("$PY" -c 'import torch; print(torch.version.cuda)')"
echo "Audio:    $AUDIO"
echo "Language: $LANG"
echo ""

TMPDIR_BENCH=$(mktemp -d)
trap 'rm -rf "$TMPDIR_BENCH"' EXIT

echo "--- Baseline (transformers) ---"
"$PY" "$DIR/benchmark_hf.py" "$AUDIO" --language "$LANG" --runs 5 | tee "$TMPDIR_BENCH/hf.txt" || true
echo ""

echo "--- nano-cohere-transcribe ---"
"$PY" "$DIR/benchmark_nano.py" "$AUDIO" --language "$LANG" --runs 5 | tee "$TMPDIR_BENCH/nano.txt"
echo ""

"$PY" - "$TMPDIR_BENCH/hf.txt" "$TMPDIR_BENCH/nano.txt" "$GPU" << 'PYEOF'
import re, sys

hf_path, nano_path, gpu = sys.argv[1:4]

def parse(text, key):
    m = re.search(rf'{key}=([0-9.]+)', text)
    return float(m.group(1)) if m else None

def parse_text(text):
    # The `{text!r}` format uses double-quotes when the payload contains `'`.
    m = re.search(r'text="((?:\\.|[^"\\])*)"', text)
    if m:
        return m.group(1)
    m = re.search(r"text='((?:\\.|[^'\\])*)'", text)
    return m.group(1) if m else None

hf = open(hf_path).read() if hf_path else ''
nano = open(nano_path).read()

def fmt(v, spec, suf=''):
    return f'{v:{spec}}{suf}' if v is not None else 'N/A'

print(f'=== Results: {gpu} ===')
print(f'{"":20} {"transformers":>20} {"nano-cohere":>20} {"Speedup":>10}')
print('-' * 73)

hf_rtf = parse(hf, 'RTF')
nano_rtf = parse(nano, 'RTF')
speedup = nano_rtf / hf_rtf if (hf_rtf and nano_rtf) else None
print(f'{"RTFx":20} {fmt(hf_rtf,">19.1f","x")} {fmt(nano_rtf,">19.1f","x")} {fmt(speedup,">9.2f","x")}')

hf_t = parse(hf, 'time_s'); hf_s = parse(hf, 'std')
nano_t = parse(nano, 'time_s'); nano_s = parse(nano, 'std')
def tfmt(m, s):
    if m is None: return 'N/A'.rjust(20)
    return (f'{m:.4f}s ±{s:.4f}' if s is not None else f'{m:.4f}s').rjust(20)
print(f'{"Inference time":20} {tfmt(hf_t,hf_s)} {tfmt(nano_t,nano_s)}')

hf_c = parse(hf, 'cold_s'); nano_c = parse(nano, 'cold_s')
def cfmt(v):
    return (f'{v:.2f}s').rjust(20) if v is not None else 'N/A'.rjust(20)
print(f'{"Cold start":20} {cfmt(hf_c)} {cfmt(nano_c)}')

a = parse(hf, 'audio_s') or parse(nano, 'audio_s')
print(f'{"Audio duration":20} {fmt(a,">19.2f","s")}')

hf_text = parse_text(hf); nano_text = parse_text(nano)
match = 'MATCH' if (hf_text and nano_text and hf_text == nano_text) else 'DIFFER'
print()
print(f'Transcripts: {match}')
if hf_text:
    print(f'  transformers: {hf_text!r}')
if nano_text:
    print(f'  nano-cohere:  {nano_text!r}')
PYEOF
