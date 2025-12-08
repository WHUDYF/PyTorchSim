import os
import sys
import argparse
import copy
import torch
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaForCausalLM

def test_result(name, out, ref, rtol=1e-4, atol=1e-4):
    if torch.allclose(out.cpu(), ref.cpu(), rtol=rtol, atol=atol):
        msg = f"|{name} Test Passed|"
        print("-" * len(msg)); print(msg); print("-" * len(msg))
    else:
        msg = f"|{name} Test Failed|"
        print("-" * len(msg)); print(msg); print("-" * len(msg))
        diff = (out.cpu() - ref.cpu()).abs().max().item()
        print("device out:", out.detach().cpu())
        print("cpu ref  :", ref.detach().cpu())
        print(f"Max abs diff: {diff}")
        sys.exit(1)

@torch.no_grad()
def run_custom_llama_test(
    device,
    batch=1,
    seq_len=32,
    dtype="float32",
    rtol=1e-3,
    atol=1e-3,
    max_new_tokens=16,
):
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    torch_dtype = dtype_map.get(dtype, torch.float32)

    cfg = LlamaConfig(
        _name_or_path="custom-llama",
        architectures=["LlamaForCausalLM"],
        attention_bias=False,
        attention_dropout=0.0,
        bos_token_id=1,
        eos_token_id=2,
        hidden_act="silu",
        hidden_size=4096,
        initializer_range=0.02,
        intermediate_size=11008,
        max_position_embeddings=4096,
        mlp_bias=False,
        model_type="llama",
        num_attention_heads=32,
        num_hidden_layers=1,
        num_key_value_heads=32,
        pretraining_tp=1,
        rms_norm_eps=1e-06,
        rope_scaling=None,
        rope_theta=10000.0,
        tie_word_embeddings=True,
        torch_dtype=dtype,
        transformers_version="4.43.4",
        use_cache=True,
        vocab_size=8192,
    )

    print("Building LlamaForCausalLM from custom config (random init).")
    base_model = LlamaForCausalLM(cfg).eval()
    cpu_model  = copy.deepcopy(base_model).eval()

    # dtype & device 세팅
    cpu_model.to(dtype=torch_dtype, device="cpu")
    model = base_model.to(dtype=torch_dtype, device=device)

    # ---- 입력 텐서 (랜덤 ids) ----
    g = torch.Generator().manual_seed(0)
    vocab = cfg.vocab_size
    input_ids_cpu = torch.randint(low=0, high=vocab, size=(batch, seq_len), generator=g, dtype=torch.long)
    attn_mask_cpu = torch.ones_like(input_ids_cpu, dtype=torch.long)

    input_ids_dev = input_ids_cpu.to(device)
    attn_mask_dev = attn_mask_cpu.to(device)

    # ---- forward comparison (compile vs CPU baseline) ----
    print("Compiling model with torch.compile(...)")
    compiled = torch.compile(model, dynamic=False)

    logits_cpu = cpu_model(input_ids=input_ids_cpu, attention_mask=attn_mask_cpu).logits
    logits_dev = compiled(input_ids=input_ids_dev, attention_mask=attn_mask_dev).logits

    test_result("Custom Llama forward(logits)", logits_dev, logits_cpu, rtol=rtol, atol=atol)
    print("Max diff >", (logits_dev.detach().cpu() - logits_cpu.detach().cpu()).abs().max().item())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Custom Llama (random weights, no tokenizer)")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=32)
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--max_new_tokens", type=int, default=16)
    args = parser.parse_args()

    sys.path.append(os.environ.get("PYTORCHSIM_ROOT_PATH", "/workspace/PyTorchSim"))
    from Scheduler.scheduler import PyTorchSimRunner
    module = PyTorchSimRunner.setup_device()
    device = module.custom_device()
    #test_triu(device, size=(32, 128), diagonal=1)
    torch.compiler.is_compiling = lambda: True # FIXME. How to fix this?
    run_custom_llama_test(
        device=device,
        batch=args.batch,
        seq_len=args.seq_len,
        dtype=args.dtype,
        rtol=args.rtol,
        atol=args.atol,
    )
