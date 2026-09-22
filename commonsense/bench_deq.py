
import os, sys, time, torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from olmoe_modification.modeling_olmoe import OlmoeAdapterForCausalLM
from olmoe_modification.configuration_olmoe import OlmoeAdapterConfig
from utils import init_trainable_parameters, convert_trainable_parameters

BASE='allenai/OLMoE-1B-7B-0924'; H=88; B,L=16,256
torch.ones(1, device='cuda')

def build(n_iter, cross):
    cfg = OlmoeAdapterConfig.from_pretrained(BASE)
    cfg.deq_routing_adapter=True
    cfg.deq_routing_adapter_args={"hidden_dim":H,"beta":1.0,"n_iter":n_iter,"tol":1e-3,
                                  "expert_aware_gate":False,"input_form":"concat","cross_layer":cross}
    m = OlmoeAdapterForCausalLM.from_pretrained(BASE, config=cfg,
            torch_dtype=torch.bfloat16, device_map={'':0})
    init_trainable_parameters(m)
    convert_trainable_parameters(m, ['deq_routing_adapter'])
    m.config.use_cache=False
    return m

def measure(n_iter, cross):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    m = build(n_iter, cross)
    base_mem = torch.cuda.max_memory_allocated()/2**30
    ids = torch.randint(0, 50000, (B, L), device='cuda')
    torch.cuda.reset_peak_memory_stats()
    for _ in range(3):  # warmup
        m(input_ids=ids, labels=ids).loss.backward(); m.zero_grad(set_to_none=True)
    torch.cuda.synchronize(); t0=time.time()
    for _ in range(10):
        m(input_ids=ids, labels=ids).loss.backward(); m.zero_grad(set_to_none=True)
    torch.cuda.synchronize(); train_ms=(time.time()-t0)/10*1000
    train_peak = torch.cuda.max_memory_allocated()/2**30
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for _ in range(3): m(input_ids=ids)
        torch.cuda.synchronize(); t0=time.time()
        for _ in range(10): m(input_ids=ids)
        torch.cuda.synchronize(); infer_ms=(time.time()-t0)/10*1000
    infer_peak = torch.cuda.max_memory_allocated()/2**30
    del m; torch.cuda.empty_cache()
    return base_mem, train_peak, train_ms, infer_peak, infer_ms

print('%8s %7s %12s %12s %12s %12s'%('n_iter','xlayer','weights GB','train peak GB','train ms/step','infer ms/step'))
for cross in [False, True]:
    for n in [0,1,3,5]:
        try:
            bm,tp,tm,ip,im = measure(n, cross)
            print('%8d %7s %12.2f %12.2f %12.1f %12.1f'%(n, 'on' if cross else 'off', bm, tp, tm, im))
        except RuntimeError as e:
            print('%8d %7s  failed: %s'%(n,'on' if cross else 'off',str(e)[:60]))
