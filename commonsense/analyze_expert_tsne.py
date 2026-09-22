
import argparse, json, os, sys, types
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import commonsense_evaluate as ce

TASKS = ["boolq", "piqa", "social_i_qa", "hellaswag", "winogrande", "ARC-Challenge", "ARC-Easy", "openbookqa"]

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True)
p.add_argument("--base_model", default="allenai/OLMoE-1B-7B-0924")
p.add_argument("--layers", default="3,7,11,15", help="layers at which expert representations are captured")
p.add_argument("--per_task", type=int, default=40, help="prompts per task")
p.add_argument("--batch_size", type=int, default=16)
p.add_argument("--cutoff", type=int, default=256)
p.add_argument("--tok_per_batch", type=int, default=48, help="no-drift tokens stored per batch per layer")
p.add_argument("--n_experts_plot", type=int, default=8, help="t-SNE uses the most frequently activated experts of the layer")
p.add_argument("--max_points", type=int, default=6000)
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()
torch.manual_seed(a.seed); np.random.seed(a.seed)
CAP_LAYERS = [int(x) for x in a.layers.split(",")]

margs = types.SimpleNamespace(base_model=a.base_model, peft_model=f"checkpoints/{a.ckpt}")
tokenizer, model = ce.load_model(margs)
model.eval()
blocks = [l.mlp for l in model.model.layers]
L = len(blocks); K = blocks[0].top_k
adapters = [b.deq_routing_adapter for b in blocks]
assert all(ad is not None and ad is not False for ad in adapters), "checkpoint does not carry a DEQ adapter in every layer"

# ---------- hooks ----------
state = {"topk": [None] * L}
def mk_gate_hook(i):
    def hook(mod, inp, out):
        state["topk"][i] = out.float().topk(K, dim=-1).indices
    return hook
for i, b in enumerate(blocks):
    b.gate.register_forward_hook(mk_gate_hook(i))

def wrap_F(ad):
    orig = ad._F
    def f(E, anchor, x, base_weights, m_prev=None):
        out = orig(E, anchor, x, base_weights, m_prev)
        ad._cap = (anchor, out, base_weights)
        return out
    ad._F = f
for i in CAP_LAYERS:
    wrap_F(adapters[i])

def set_adapters(on):
    for b, ad in zip(blocks, adapters):
        b.deq_routing_adapter = ad if on else None

prompts = []
for t in TASKS:
    data = json.load(open(f"dataset/{t}/test.json"))[: a.per_task]
    prompts += [ce.generate_prompt(d.get("instruction"), d.get("input")) for d in data]
print(f"##### {len(prompts)} prompts from {len(TASKS)} tasks #####")

drift = {k: np.zeros(L) for k in ("exact", "jaccard", "top1")}
n_tok = 0
store = {i: {"E0": [], "Es": [], "eid": [], "w": []} for i in CAP_LAYERS}
dev = next(model.parameters()).device

with torch.no_grad():
    for s in range(0, len(prompts), a.batch_size):
        enc = tokenizer(prompts[s:s + a.batch_size], return_tensors="pt", padding=True,
                        truncation=True, max_length=a.cutoff).to(dev)
        keep = enc["attention_mask"].reshape(-1).bool()

        set_adapters(False); model(**enc, use_cache=False)
        base_tk = [t.clone() for t in state["topk"]]
        set_adapters(True);  model(**enc, use_cache=False)
        tuned_tk = state["topk"]

        n_tok += int(keep.sum())
        for i in range(L):
            b_, t_ = base_tk[i][keep], tuned_tk[i][keep]                       # (T',k)
            inter = (b_.unsqueeze(2) == t_.unsqueeze(1)).any(2).sum(1).float()
            exact = inter == K
            drift["exact"][i] += exact.sum().item()
            drift["jaccard"][i] += (inter / (2 * K - inter)).sum().item()
            drift["top1"][i] += (b_[:, 0] == t_[:, 0]).sum().item()
            if i in CAP_LAYERS:
                anchor, Es, w = adapters[i]._cap
                idx = torch.nonzero(keep, as_tuple=False).squeeze(1)[exact]
                if len(idx) > a.tok_per_batch:
                    idx = idx[torch.randperm(len(idx), device=idx.device)[: a.tok_per_batch]]
                store[i]["E0"].append(anchor[idx].float().cpu())
                store[i]["Es"].append(Es[idx].float().cpu())
                store[i]["eid"].append(tuned_tk[i][idx].cpu())
                store[i]["w"].append(w[idx].float().cpu())
        print(f"  batch {s // a.batch_size + 1}: tokens so far {n_tok}", flush=True)

for k in drift: drift[k] = (drift[k] / n_tok).tolist()
print("\n##### routing drift per layer (1.0 = identical to the base model) #####")
print(f"{'layer':>5}{'exact-set':>11}{'jaccard':>9}{'top-1':>8}")
for i in range(L):
    print(f"{i:>5}{drift['exact'][i]:>11.4f}{drift['jaccard'][i]:>9.4f}{drift['top1'][i]:>8.4f}")

from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
out, stats = {}, {"ckpt": a.ckpt, "n_tokens": n_tok, "drift": drift, "layers": {}}
for i in CAP_LAYERS:
    E0 = torch.cat(store[i]["E0"]); Es = torch.cat(store[i]["Es"]); eid = torch.cat(store[i]["eid"])
    T_, k_, d_ = E0.shape
    def mean_pair_cos(E):
        U = torch.nn.functional.normalize(E, dim=-1); S = U.sum(1)
        return ((S.pow(2).sum(-1) - k_) / (k_ * (k_ - 1))).mean().item()
    rel = ((Es - E0).norm(dim=-1) / (E0.norm(dim=-1) + 1e-8)).mean().item()
    E0f, Esf, idf = E0.reshape(-1, d_), Es.reshape(-1, d_), eid.reshape(-1)
    top = torch.bincount(idf, minlength=blocks[i].num_experts).topk(a.n_experts_plot).indices
    m = torch.isin(idf, top)
    sel = torch.nonzero(m).squeeze(1)
    if len(sel) > a.max_points:
        sel = sel[torch.randperm(len(sel))[: a.max_points]]
    X0, Xs, y = E0f[sel].numpy(), Esf[sel].numpy(), idf[sel].numpy()
    sil0 = silhouette_score(X0, y, metric="cosine"); sils = silhouette_score(Xs, y, metric="cosine")
    Z = TSNE(n_components=2, perplexity=30, init="pca", metric="cosine", random_state=a.seed
             ).fit_transform(np.concatenate([X0, Xs], 0))
    n = len(sel)
    out[f"L{i}_Z0"], out[f"L{i}_Zs"], out[f"L{i}_y"] = Z[:n], Z[n:], y
    stats["layers"][str(i)] = {"tokens_no_drift": T_, "points": int(n), "experts": top.tolist(),
                               "pair_cos_E0": mean_pair_cos(E0), "pair_cos_Estar": mean_pair_cos(Es),
                               "silhouette_E0": float(sil0), "silhouette_Estar": float(sils),
                               "rel_change": rel}
    print(f"layer {i}: no-drift tokens {T_}, pair-cos {stats['layers'][str(i)]['pair_cos_E0']:.4f} -> "
          f"{stats['layers'][str(i)]['pair_cos_Estar']:.4f}, silhouette {sil0:.4f} -> {sils:.4f}, "
          f"|E*-E0|/|E0| = {rel:.4f}", flush=True)

os.makedirs("../figures", exist_ok=True)
np.savez_compressed(f"../figures/expert_tsne_{a.ckpt}.npz", **out)
json.dump(stats, open(f"../figures/expert_tsne_{a.ckpt}.json", "w"), indent=1)
print(f"##### saved ../figures/expert_tsne_{a.ckpt}.npz / .json #####")
