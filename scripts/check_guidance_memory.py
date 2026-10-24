"""One real 1024px backward pass and one attention update on the local GPU."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import time
import torch
from visual_anagrams.iris import load_model, resolve_model, read_config, PromptEncoding
from visual_anagrams.optimization import gradient_memory
from visual_anagrams.advanced_sampling import optimize_overlap
from visual_anagrams.method_options import MethodOptions
from visual_anagrams.views import get_views

torch.set_num_threads(6)
root = Path("results/quality_pilot_20261024")
saved = torch.load(root / "prompt-encodings.pt", weights_only=True)
directory = resolve_model()
model = load_model(directory, read_config(directory), "cuda")
positive = PromptEncoding(saved["embeddings"][:2].cuda().clone(), saved["mask"][:2].cuda().clone())
image = torch.randn(1, 3, 1024, 1024, device="cuda", generator=torch.Generator(device="cuda").manual_seed(0))
torch.cuda.reset_peak_memory_stats()
start = time.perf_counter()
with torch.enable_grad(), torch.autocast("cuda", dtype=torch.bfloat16), gradient_memory(model):
    candidate = image.clone().requires_grad_(True)
    velocity = model(candidate, torch.tensor([800.], device="cuda"), positive.embeddings[:1], y_mask=positive.mask[:1]).x
    gradient, = torch.autograd.grad(velocity.float().square().mean(), candidate)
    assert torch.isfinite(gradient).all() and gradient.square().mean() > 0
torch.cuda.synchronize()
result = dict(full_backward_seconds=time.perf_counter()-start,
              full_backward_peak_gib=torch.cuda.max_memory_allocated()/2**30,
              gradient_rms=float(gradient.square().mean().sqrt()))
print(json.dumps(result), flush=True)
del candidate, velocity, gradient
from transformers import AutoTokenizer
from visual_anagrams.optimization import subject_token_indices
from visual_anagrams.iris import TEXT_ID, TEXT_REVISION
tokenizer=AutoTokenizer.from_pretrained(TEXT_ID,revision=TEXT_REVISION)
positions=subject_token_indices(tokenizer,["a watercolor painting of an owl","a watercolor painting of a dog"],["owl","dog"])
start=time.perf_counter()
torch.cuda.reset_peak_memory_stats()
with torch.autocast("cuda", dtype=torch.bfloat16):
    updated, extra = optimize_overlap(model,image,.8,positive,get_views(["identity","rotate_180"]),positions,MethodOptions(method="anagram_mtl"))
assert torch.isfinite(updated).all()
result.update(attention_seconds=time.perf_counter()-start,attention_peak_gib=torch.cuda.max_memory_allocated()/2**30,
              attention_change_rms=float((updated-image).square().mean().sqrt()),attention=extra,positions=positions)
Path('.research/guidance-memory.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result),flush=True)
