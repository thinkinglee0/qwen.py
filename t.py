from qwen import QwenModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import inspect


def sampling(tok, last_logit):
    output_ids = last_logit.argmax()
    return tok.decode(output_ids.item())

model_dir = "../qwen2.5-0.5b"
tok = AutoTokenizer.from_pretrained(model_dir)

input_str = "The capital of France is"
input_ids = tok(input_str, return_tensors="pt").input_ids
print("input_ids: ", input_ids.shape)


model = QwenModel(model_dir)

# Example usage of the model
target_output = model.forward(input_ids)
print("target_output logits shape: ", target_output.logits.shape)
print("next token from qwen.py model: ", sampling(tok, target_output.logits[0, -1]))    # expect " Paris"

# final_norm_hidden_states = model.forward(input_ids)

cap = {}

import transformers.models.qwen2.modeling_qwen2 as qwen2_modeling

# original_rope = qwen2_modeling.apply_rotary_pos_emb
# def patched_rope(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
#     cap.setdefault("all_q_before_rope", []).append(q.detach().clone())
#     cap.setdefault("all_k_before_rope", []).append(k.detach().clone())
#     cap.setdefault("sin", []).append(sin.detach().clone())
#     cap.setdefault("cos", []).append(cos.detach().clone())
#     q_out, k_out = original_rope(q, k, cos, sin, position_ids, unsqueeze_dim)
#     cap.setdefault("all_q_after_rope", []).append(q_out.detach().clone())
#     cap.setdefault("all_k_after_rope", []).append(k_out.detach().clone())
#     return q_out, k_out
# qwen2_modeling.apply_rotary_pos_emb = patched_rope

original_repeat_kv = qwen2_modeling.repeat_kv
def patched_repeat_kv(k_or_v_states, n_rep):
    cap.setdefault("k_or_v_states_before_repeat_kv", []).append(k_or_v_states.detach().clone())
    k_or_v_states_out = original_repeat_kv(k_or_v_states, n_rep)
    cap.setdefault("k_or_v_states_after_repeat_kv", []).append(k_or_v_states_out.detach().clone())
    return k_or_v_states_out
qwen2_modeling.repeat_kv = patched_repeat_kv




ref = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32, attn_implementation="eager")
ref.eval()

def save_i(name):
    def pre_hook(module, input):
        val = input[0] if isinstance(input, tuple) else input
        cap[name] = val.detach().clone()    # clone instantly, avoiding modified by following flow
    return pre_hook
def save_o(name):
    def hook(module, input, output):
        cap[name] = (output[0] if isinstance(output, tuple) else output).detach().clone()
    return hook
# ref.model.embed_tokens.register_forward_hook(save_o("embed_tokens_o"))
# ref.model.layers[0].self_attn.q_proj.register_forward_pre_hook(save_i("L0.attn_q_i"))
# ref.model.layers[0].self_attn.q_proj.register_forward_hook(save_o("L0.attn_q_o"))
# ref.model.layers[0].self_attn.k_proj.register_forward_hook(save_o("L0.attn_k_o"))
# ref.model.layers[0].self_attn.v_proj.register_forward_hook(save_o("L0.attn_v_o"))
# ref.model.layers[0].post_attention_layernorm.register_forward_pre_hook(save_i("L0.post_attention_layernorm_i"))
# ref.model.layers[0].post_attention_layernorm.register_forward_hook(save_o("L0.post_attention_layernorm_o"))
# ref.model.layers[0].mlp.gate_proj.register_forward_hook(save_o("L0.mlp.gate_proj_o"))
# ref.model.layers[0].mlp.up_proj.register_forward_hook(save_o("L0.mlp.up_proj_o"))
# ref.model.layers[0].mlp.down_proj.register_forward_hook(save_o("L0.mlp.down_proj_o"))
# ref.model.norm.register_forward_hook(save_o("final_norm_o"))
ref.lm_head.register_forward_hook(save_o("lm_head_o"))



with torch.no_grad():
    ref_output = ref.forward(input_ids)
# print(inspect.getfile(type(ref_output)))
# print(inspect.getsource(type(ref_output)))
print("ref_output logits shape: ", ref_output.logits.shape)
print("next token from reference model: ", sampling(tok, ref_output.logits[0, -1]))    # expect " Paris"

def compare(target, ref, name="", rtol=1e-3, atol=1e-3):
    try:
        torch.testing.assert_close(
            target, ref,
            rtol=rtol, atol=atol,
            check_dtype=False, check_device=False,   # 只看数值，dtype/device 不同先 promote 再比
        )
        print(f"[PASS] {name}, shape={tuple(target.shape)}")
        return True
    except AssertionError as e:
        # assert_close 的 message 已含 shape mismatch / Mismatched elements 比例 /
        # Greatest absolute & relative difference + 各自 index
        print(f"[FAIL] {name}\n{e}")
        return False

# compare(output, cap["embed_tokens_o"])
# compare(query_states_before_reshape, cap["L0.attn_q_o"])
# compare(key_states_before_reshape, cap["L0.attn_k_o"])

# compare(query_states, cap["all_q_after_rope"][0])  # only compare layer 0
# compare(key_states, cap["all_k_after_rope"][0])  # only compare layer 0
# compare(cos, cap["cos"][0])  # only compare layer 0
# compare(sin, cap["sin"][0])  # only compare layer 0

# print(len(cap["k_or_v_states_before_repeat_kv"]))  #  num_layers = 24
# print(len(cap["k_or_v_states_after_repeat_kv"]))  #  num_layers = 24

# compare(key_states, cap["k_or_v_states_after_repeat_kv"][0])  # k of the first layer
# compare(value_states, cap["k_or_v_states_after_repeat_kv"][1])  # v of the first layer

# compare(gate, cap["L0.mlp.gate_proj_o"])
# compare(up, cap["L0.mlp.up_proj_o"])
# compare(hidden_states, cap["L0.mlp.down_proj_o"])

# compare(final_norm_hidden_states, cap["final_norm_o"])

print(ref.dtype)  # confirm the precision of the reference model
compare(target_output.logits, cap["lm_head_o"], "logits vs ref.lm_head")
compare(ref_output.logits, cap["lm_head_o"], "ref.logits vs ref.lm_head")
compare(target_output.logits, ref_output.logits, "target.logits vs ref.logits")
