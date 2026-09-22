# Fine-Tune How Experts Coordinate: Consensus-Anchored Adaptation of Pre-trained Mixture-of-Experts

Anonymous code release for the ICLR 2026 submission of the same title. The method is
referred to as **CAFT** (Consensus-Anchored Fine-Tuning) below.

This repository fine-tunes a **frozen** pretrained MoE model (experts, backbone
and router all frozen) by training one small shared operator per MoE layer that
refines the outputs of the top-*k* active experts. Each expert's refinement is
conditioned on a *coordination signal* computed from its co-active peers
(within-layer consensus), optionally carried across depth (cross-layer
consensus momentum). The refined representations are defined as the fixed
point of an anchored coupling map and trained with a one-step (phantom)
gradient, so memory is constant in the number of solver iterations.

In the code CAFT is enabled with `--deq_routing_adapter True`; the class
implementing it is `*DEQRoutingAdapter` in each `*_modification/modeling_*.py`.

## Repository layout

```
olmoe_modification/     OLMoE-1B-7B  model + adapter classes (CAFT operator: OlmoeDEQRoutingAdapter in modeling_olmoe.py)
mixtral_modification/   Mixtral-8x7B  model + adapter classes
qwen3_modification/     Qwen3-30B-A3B model + adapter classes (needs transformers >= 4.51)
commonsense/            training / evaluation on Commonsense170K -> 8 commonsense benchmarks
math/                   training / evaluation on Math50K -> 6 arithmetic benchmarks
env.sh                  HF cache + NO_TORCH_COMPILE, sourced by every launch script
figures/                architecture figure source
```

Each task directory has its own `finetune.py`, `*_evaluate.py`, `utils.py`
(identity initialisation of the operator lives in `utils.py`), shell launch
scripts, and PBS wrappers (`qsub.*.pbs.sh`; adapt or ignore for your scheduler).

## Setup

Two conda environments are used because Qwen3-MoE requires a newer
`transformers` than OLMoE/Mixtral were developed against:

| env | backbones | key packages |
|---|---|---|
| `ml_env`   | OLMoE, Mixtral | python 3.8, torch 2.4.1, transformers 4.46.3, safetensors, scikit-learn |
| `qwen3moe` | Qwen3-30B-A3B  | python 3.11, torch 2.4.1, transformers 4.51.3 |

```bash
source env.sh                      # sets HF_HOME etc.; override HF_CACHE_ROOT for a different disk
export NO_TORCH_COMPILE=1          # torch.compile hangs on first forward under device_map='auto'
```

**Data.** Download `commonsense_170k.json`, `math_50k.json` and the `dataset/`
evaluation folders from the LLM-Adapters release and place them in
`commonsense/` and `math/` respectively (they are git-ignored).

## Training

All arguments are documented in `commonsense/finetune.py`. The ones that
define the method:

| flag | meaning |
|---|---|
| `--deq_routing_adapter True` | enable CAFT (experts / router stay frozen) |
| `--deq_hidden_dim h` | bottleneck width; trainable params = 3·d·h per MoE layer (`concat`) |
| `--deq_input_form concat\|eonly\|diff` | operator input `[E_i; c_i]` (default) / `[E_i]` / `[c_i − E_i]` |
| `--deq_n_iter n` | fixed-point iterations (`n=0` = no feedback, one feed-forward application) |
| `--deq_cross_layer True` | cross-layer consensus momentum `m^l = a·m^{l-1} + (1−a)·c^l` |
| `--deq_expert_aware_gate False` | refine-only (used for all reported results) |
| `--deq_beta 1.0` | step size (redundant with the operator's output projection; kept at 1) |

Environment variables read by `finetune.py` / `utils.py`:

| variable | default | effect |
|---|---|---|
| `DEQ_SEED` | 42 | seed for data split, shuffling and `TrainingArguments` |
| `DEQ_ALPHA_INIT` | −0.5 | raw initial value of the per-layer cross-layer coefficient (a = tanh(raw)) |
| `DEQ_ALPHA_LR` | unset | separate learning rate for the cross-layer coefficient |

### OLMoE-1B-7B (2 GPUs, DDP), commonsense

```bash
cd commonsense
# main configuration: h=88, cross-layer, n=3   (8.65M trainable)
torchrun --nproc_per_node=2 finetune.py \
  --base_model allenai/OLMoE-1B-7B-0924 --data_path commonsense_170k.json \
  --output_dir ./checkpoints/OLMoE-1B-7B.deq_h88_xlayer_refineonly_n3_lr2e-4 \
  --batch_size 32 --micro_batch_size 16 --num_epochs 3 --learning_rate 2e-4 \
  --cutoff_len 256 --val_set_size 120 --eval_step 80 --save_step 80 \
  --deq_routing_adapter True --deq_hidden_dim 88 --deq_beta 1.0 --deq_n_iter 3 --deq_tol 1e-3 \
  --deq_expert_aware_gate False --deq_input_form concat --deq_cross_layer True
```

`OLMoE-1B-7B.ddp.sh` wraps the same command (`bash OLMoE-1B-7B.ddp.sh deq 2e-4 88 3 concat on`).

### Mixtral-8x7B / Qwen3-30B-A3B (2 GPUs, model-parallel `device_map='auto'`)

```bash
cd commonsense
bash Mixtral-8x7B.deq.sh 2e-4 8 3 concat on        # lr, h, n_iter, input_form, cross-layer
# Qwen3 (activate the qwen3moe env first): same flags as OLMoE with
#   --base_model Qwen/Qwen3-30B-A3B-Base --batch_size 16 --micro_batch_size 16 --deq_hidden_dim 29
```

Parameter budgets used in the paper: OLMoE `h=88` = 8.65M; Mixtral `h=8` = 3.15M; Qwen3 `h=29` = 8.65M. Trainable counts are printed at start-up
(`Print trainable params: ...`).

## Evaluation

```bash
cd commonsense
bash eval_deq.sh OLMoE-1B-7B.deq_h88_xlayer_refineonly_n3_lr2e-4 0 1      # two GPUs, 4 tasks each
cd ../math
bash eval_deq.sh Mixtral-8x7B.math.deq_h8_xlayer_refineonly_n3_lr2e-4 0 1
```

Per-example outputs are written to `experiment/<run>-<task>.json`; accuracy is
the mean of the `flag` field. The base model is chosen from the checkpoint name.

Evaluation-time controls (set as environment variables before `eval_deq.sh`):

| variable | effect |
|---|---|
| `DEQ_EVAL_NITER=n` | override the number of fixed-point iterations at inference (Table: iterations at inference time) |
| `DEQ_EVAL_INTERVENE=zero_m\|shuffle_m\|alpha0\|alpha_flip` | interventions on the cross-layer state (m := 0, m permuted over tokens, a := 0, a := −a) |

Results are written under a suffixed tag (`.evn<n>` / `.iv_<name>`) so they do
not overwrite the checkpoint's normal evaluation.

## Analysis scripts

| script | purpose |
|---|---|
| `commonsense/bench_deq.py` | peak training memory and step time vs. `n_iter` (single GPU, synthetic batch) |
| `commonsense/analyze_expert_tsne.py` | per-layer routing agreement with the base model, and expert-representation statistics on no-drift tokens |

Training logs the solver's first/last relative residual, the refinement
magnitude ‖E*−E⁰‖/‖E⁰‖, the mean pairwise cosine among refined experts and
the cross-layer coefficients to Weights & Biases (`deq/*` keys).

## Licenses

Model code derives from Hugging Face `transformers` (Apache 2.0). Training and
evaluation scaffolding and the benchmark data follow LLM-Adapters; see `LICENSE`
and `DATA_LICENSE` in each task directory.
