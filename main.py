import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

# 引入你的自定义模块 (请确保 tree_topology.py 已经更新为通用的 k-ary 版本)
from tree_topology import build_tree_topology, generate_tree_attention_mask, generate_position_ids
from draft_generator import generate_draft_tree
from target_verifier import verify_tree_and_accept

# ==========================================
# 🚀 中央超参数配置区 (改这里，全局自动适配)
# ==========================================
DEPTH = 4         # 树的深度 (向未来推测的步数)
BRANCH = 3           # 分叉数 (每个节点的 Top-K 候选数)
MAX_NEW_TOKENS = 50  # 最大生成长度
# ==========================================

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("老板，正在加载模型权重...")
    target_model_name = "./Model/Llama-7B-Chat-Target" 
    draft_model_name = "./Model/Llama-68M-Draft"       
    
    tokenizer = AutoTokenizer.from_pretrained(target_model_name)
    target_model = AutoModelForCausalLM.from_pretrained(target_model_name, torch_dtype=torch.float16, device_map="cuda:0", attn_implementation="eager")
    draft_model = AutoModelForCausalLM.from_pretrained(draft_model_name, torch_dtype=torch.float16, device_map="cuda:0", attn_implementation="eager")
    
    prompt = "The capital of France is Paris. The capital of Japan is"
    stop_words = ["\n\n"] 
    eos_token_id = tokenizer.eos_token_id

    # 1. 自动计算拓扑结构
    print(f"\n正在构建 {BRANCH}叉树, 深度为 {DEPTH}...")
    # 注意：这里的 build_tree_topology 需要是你更新后的通用版本
    parents, total_nodes = build_tree_topology(depth=DEPTH, branch=BRANCH)
    print(f"树的总节点数: {total_nodes} 个")

    # =====================================================================
    # 🏆 第一半场：推测解码 (Speculative Decoding)
    # =====================================================================
    print("\n" + "="*50)
    print("🚀 [上半场] 开始推测解码 (SD) 推理...")
    print("="*50)
    
    input_ids_sd = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    sd_generated_tokens = 0
    total_draft_time = 0.0
    total_verify_time = 0.0
    total_accepted_tokens_count = 0
    sd_iterations = 0
    
    while sd_generated_tokens < MAX_NEW_TOKENS:
        seq_len = input_ids_sd.shape[1]
        sd_iterations += 1
        
        # 动态生成树状掩码和位置编码
        tree_mask = generate_tree_attention_mask(parents, seq_len, dtype=target_model.dtype, device=device)
        pos_ids = generate_position_ids(parents, base_position=seq_len-1, device=device)
        
        # --- ⏱️ 计时：小模型起草 ---
        torch.cuda.synchronize() 
        t0 = time.perf_counter()
        draft_tokens = generate_draft_tree(draft_model, input_ids_sd, parents, tree_mask, pos_ids, branch=BRANCH)
        torch.cuda.synchronize() 
        t1 = time.perf_counter()
        round_draft_time = t1 - t0
        
        # --- ⏱️ 计时：大模型验证 ---
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        accepted_tokens = verify_tree_and_accept(
            target_model, input_ids_sd, draft_tokens, parents, tree_mask, pos_ids, tokenizer
        )
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        round_verify_time = t3 - t2

        round_accepted_count = len(accepted_tokens)
        decoded_accepted = tokenizer.decode(accepted_tokens)
        
        # 刹车逻辑
        is_finished = False
        if eos_token_id in accepted_tokens:
            eos_index = accepted_tokens.index(eos_token_id)
            accepted_tokens = accepted_tokens[:eos_index + 1]
            is_finished = True
        
        for stop_word in stop_words:
            if stop_word in decoded_accepted:
                is_finished = True
                break
        
        # 拼接更新主序列
        accepted_tensor = torch.tensor([accepted_tokens], dtype=torch.long, device=device)
        input_ids_sd = torch.cat([input_ids_sd, accepted_tensor], dim=1)
        sd_generated_tokens += len(accepted_tokens)
        
        # 排除第一轮预热
        if sd_iterations > 1:
            total_draft_time += round_draft_time
            total_verify_time += round_verify_time
            total_accepted_tokens_count += round_accepted_count

        if is_finished:
            break

    sd_valid_rounds = max(1, sd_iterations - 1)
    sd_total_time = total_draft_time + total_verify_time
    sd_tps = total_accepted_tokens_count / sd_total_time if sd_total_time > 0 else 0


    # =====================================================================
    # 🐢 第二半场：纯目标模型自回归推理 (Baseline)
    # =====================================================================
    print("\n" + "="*50)
    print("🐢 [下半场] 开始纯目标模型 (Baseline) 推理...")
    print("="*50)

    input_ids_base = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    baseline_generated_tokens = 0
    past_key_values_base = DynamicCache()

    # --- ⏱️ 计时：首轮 (Prefill) ---
    torch.cuda.synchronize()
    t0_base = time.perf_counter()
    
    outputs = target_model(
        input_ids=input_ids_base,
        use_cache=True,
        past_key_values=past_key_values_base,
        attn_implementation="eager"
    )
    next_token = torch.argmax(outputs.logits[0, -1, :], dim=-1).unsqueeze(0).unsqueeze(0)
    input_ids_base = torch.cat([input_ids_base, next_token], dim=1)
    baseline_generated_tokens += 1

    torch.cuda.synchronize()
    t1_base = time.perf_counter()
    baseline_first_round_time = t1_base - t0_base
    
    # --- ⏱️ 计时：后续逐字生成 (Decode) ---
    baseline_decode_total_time = 0.0
    
    while baseline_generated_tokens < MAX_NEW_TOKENS:
        torch.cuda.synchronize()
        t2_base = time.perf_counter()
        
        outputs = target_model(
            input_ids=next_token,
            use_cache=True,
            past_key_values=past_key_values_base,
            attn_implementation="eager"
        )
        next_token = torch.argmax(outputs.logits[0, -1, :], dim=-1).unsqueeze(0).unsqueeze(0)
        
        torch.cuda.synchronize()
        t3_base = time.perf_counter()
        
        baseline_decode_total_time += (t3_base - t2_base)
        input_ids_base = torch.cat([input_ids_base, next_token], dim=1)
        baseline_generated_tokens += 1
        
        if next_token.item() == eos_token_id:
            break
        decoded_current = tokenizer.decode(next_token[0])
        if any(sw in decoded_current for sw in stop_words):
            break

    baseline_decode_steps = baseline_generated_tokens - 1
    baseline_avg_decode_time = baseline_decode_total_time / baseline_decode_steps if baseline_decode_steps > 0 else 0
    baseline_tps = baseline_decode_steps / baseline_decode_total_time if baseline_decode_total_time > 0 else 0


    # =====================================================================
    # 📊 终极对决：性能对比报告
    # =====================================================================
    print("\n" + "🔥"*25)
    print("      推测解码 VS 纯自回归 终极对决")
    print("🔥"*25)
    
    print("\n【1】纯目标模型 (Baseline) 表现：")
    print(f"  • 首轮处理 (Prefill) 耗时: {baseline_first_round_time:.4f} 秒")
    print(f"  • 排除首轮后平均推理耗时: {baseline_avg_decode_time:.4f} 秒/Token")
    print(f"  • 纯 Decode 阶段吞吐量: {baseline_tps:.2f} Tokens/秒")
    print(f"  • 生成结果: \033[90m{tokenizer.decode(input_ids_base[0])}\033[0m")

    print("\n【2】推测解码 (SD) 表现：")
    print(f"  • 拓扑结构: {BRANCH}叉 {DEPTH}层 (验证节点数: {total_nodes})")
    print(f"  • 平均起草耗时: {total_draft_time/sd_valid_rounds:.4f} 秒/轮")
    print(f"  • 平均验证耗时: {total_verify_time/sd_valid_rounds:.4f} 秒/轮")
    print(f"  • 平均每轮接受: {total_accepted_tokens_count / sd_valid_rounds:.2f} Tokens/轮")
    print(f"  • 综合生成吞吐量: {sd_tps:.2f} Tokens/秒")
    print(f"  • 生成结果: \033[96m{tokenizer.decode(input_ids_sd[0])}\033[0m")

    print("\n" + "="*50)
    speedup = sd_tps / baseline_tps if baseline_tps > 0 else 0
    print(f"🚀 终极结论：推测解码带来了 \033[93m{speedup:.2f} 倍\033[0m 的真实加速！")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()