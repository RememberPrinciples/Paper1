import torch
from transformers import DynamicCache

@torch.no_grad()
def generate_draft_tree(draft_model, prefix_input_ids, parents, tree_mask, pos_ids, branch):
    """
    真正的树状草稿生成器！
    采用 BFS (广度优先) 逐层推演，完美适配传入的 branch 分叉数和 tree_mask。
    """
    seq_len = prefix_input_ids.shape[1]
    total_nodes = len(parents)
    device = prefix_input_ids.device
    
    # 存放最终的推测 Token
    draft_tokens = torch.zeros(total_nodes, dtype=torch.long, device=device)
    
    # === 阶段一：Draft 处理 Prefix (Prefill) ===
    past_key_values = DynamicCache()
    prefix_outputs = draft_model(
        input_ids=prefix_input_ids,
        use_cache=True,
        past_key_values=past_key_values,
        attn_implementation="eager"
    )
    
    # === 阶段二：逐层生成 Tree (Decode) ===
    # 第一层：基于 Root (Prefix的最后一个词) 取 Top-K
    root_logits = prefix_outputs.logits[0, -1, :] 
    _, topk_indices = torch.topk(root_logits, branch)
    
    # 找到所有第一层的节点 (它们的 parent 是 -1)
    current_layer = [i for i, p in enumerate(parents) if p == -1]
    for i, node_idx in enumerate(current_layer):
        draft_tokens[node_idx] = topk_indices[i]
        
    # 广度优先 (BFS)：一层一层往下推演
    while len(current_layer) > 0:
        # 找到下一层的所有节点 (它们的 parent 在当前层里)
        next_layer = [i for i, p in enumerate(parents) if p in current_layer]
        if not next_layer:
            break
            
        start_idx = current_layer[0]
        end_idx = current_layer[-1] + 1
        
        # 将当前层的 Token 喂给模型，用来预测下一层
        current_input_ids = draft_tokens[current_layer].unsqueeze(0)
        
        # 🛡️ 神之一手：精准切片 Tree Mask
        # 截取属于当前层的注意力掩码和位置编码
        sliced_mask = tree_mask[:, :, seq_len + start_idx : seq_len + end_idx, :seq_len + end_idx]
        current_pos_ids = pos_ids[:, start_idx : end_idx]
        
        outputs = draft_model(
            input_ids=current_input_ids,
            attention_mask=sliced_mask,
            position_ids=current_pos_ids,
            past_key_values=past_key_values,
            use_cache=True,
            attn_implementation="eager"
        )
        
        # 为当前层的每个节点，分别预测它的 Top-K 个孩子
        for i, parent_node_idx in enumerate(current_layer):
            logits = outputs.logits[0, i, :]
            _, topk_indices = torch.topk(logits, branch) 
            
            # 把生成的候选词，精准对接到对应的孩子节点上
            children = [n for n, p in enumerate(parents) if p == parent_node_idx]
            for j, child_node_idx in enumerate(children):
                if j < len(topk_indices): # 安全校验
                    draft_tokens[child_node_idx] = topk_indices[j]
                
        current_layer = next_layer
        
    return draft_tokens