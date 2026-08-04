"""Model loading, in ONE place.

`load_real`/`load_tiny` began life inside the calibration probe and were
imported from there by every later script -- which meant the main experiment
imported from a diagnostic. They live here so the dependency points the right
way: probes and the run loop both depend on loading, never on each other.

Greedy decoding is set at load time, not per-call: every consumer of these
models relies on generation being a deterministic function of (problem,
condition), and a loader that forgot to set it would corrupt every paired
comparison downstream.
"""
import torch

MODEL = "Qwen/Qwen3-4B"


def load_real(device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    m = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16).to(device).eval()
    m.generation_config.do_sample = False
    m.generation_config.temperature = None
    m.generation_config.top_p = m.generation_config.top_k = None
    return m, tok


def load_tiny(device):
    """A randomly-initialised Qwen3 built from config: no weights, no
    download, no GPU. The numbers are meaningless but the CONTRACT is
    identical, which is what the weightless smoke tests and CI rely on."""
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=256, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=36, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=16,
                      max_position_embeddings=4096, pad_token_id=0)
    m = Qwen3ForCausalLM(cfg).to(device).eval()
    g = torch.Generator().manual_seed(7)
    with torch.no_grad():                       # trained models have a real gain
        m.model.norm.weight.copy_(
            torch.rand(cfg.hidden_size, generator=g) * 2 + 0.25)
    m.generation_config.do_sample = False
    return m, None
