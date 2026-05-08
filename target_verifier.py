import torch
from transformers import DynamicCache

def extract_paths_from_parents(parents):
    paths = []
    leaf_nodes = [i for i in range(len(parents)) if i not in parents]
    for leaf in leaf_nodes:
        path = []
        curr = leaf
        while curr != -1:
            path.append(curr)
            curr = parents[curr]
        paths.append(path[::-1])
    return paths

@torch.no_grad()
def verify_tree_and_accept(target_model, prefix_input_ids, draft_tokens, parents, tree_mask, pos_ids, tokenizer):
    """大模型双阶段验证（自带硬核 Debug 日志）"""
    
    # === 阶段一：处理 Prefix ===
    past_key_values = DynamicCache() 
    prefix_outputs = target_model(
        input_ids=prefix_input_ids,
        use_cache=True,
        past_key_values=past_key_values,
        attn_implementation="eager"
    )
    root_target_token = torch.argmax(prefix_outputs.logits[0, -1, :], dim=-1)

    # === 阶段二：处理 Draft Tree ===
    draft_input_ids = draft_tokens.unsqueeze(0)
    seq_len = prefix_input_ids.shape[1]
    sliced_tree_mask = tree_mask[:, :, seq_len:, :]
    
    tree_outputs = target_model(
        input_ids=draft_input_ids,
        attention_mask=sliced_tree_mask,
        position_ids=pos_ids,
        past_key_values=past_key_values,
        use_cache=True,
        attn_implementation="eager"
    )
    tree_target_tokens = torch.argmax(tree_outputs.logits[0], dim=-1)

    # === 🕵️ 核心监控日志（只看根节点的直接子节点，也就是树的第一层） ===
    print("\n" + "="*50)
    print("🕵️  [深度核对日志] Target vs Draft 第一轮交锋")
    
    target_expects_char = tokenizer.decode([root_target_token.item()])
    print(f"🎯 Target 模型看完 Prompt 后，内心想说的第一个词是: \033[92m'{target_expects_char}'\033[0m (ID: {root_target_token.item()})")
    print("-" * 50)
    
    # 找出所有第一层节点 (parent == -1)
    layer_1_nodes = [i for i, p in enumerate(parents) if p == -1]
    for i, node_idx in enumerate(layer_1_nodes):
        draft_char = tokenizer.decode([draft_tokens[node_idx].item()])
        match_status = "✅ 匹配成功！" if root_target_token == draft_tokens[node_idx] else "❌ 产生分歧！"
        print(f"🌿 Draft 树分支 {i+1} 猜测的词是: \033[93m'{draft_char}'\033[0m (ID: {draft_tokens[node_idx].item()}) -> {match_status}")
    print("="*50 + "\n")
    # ===============================================================

    # === 阶段三：路径验证逻辑 ===
    all_paths = extract_paths_from_parents(parents)
    best_path_tokens = []
    best_path_indices = []

    for path in all_paths:
        accepted_tokens = []
        accepted_indices = []
        for node_idx in path:
            parent_idx = parents[node_idx]
            
            if parent_idx == -1:
                expected_token = root_target_token
            else:
                expected_token = tree_target_tokens[parent_idx]

            if expected_token == draft_tokens[node_idx]:
                accepted_tokens.append(draft_tokens[node_idx].item())
                accepted_indices.append(node_idx)
            else:
                break 
                
        if len(accepted_tokens) > len(best_path_tokens):
            best_path_tokens = accepted_tokens
            best_path_indices = accepted_indices
            
    # === 阶段四：提取额外的 Bonus Token ===
    if not best_path_indices:
        extra_token = root_target_token.item()
    else:
        last_accepted_node_idx = best_path_indices[-1]
        extra_token = tree_target_tokens[last_accepted_node_idx].item()
        
    final_accepted_tokens = best_path_tokens + [extra_token]
    
    return final_accepted_tokens