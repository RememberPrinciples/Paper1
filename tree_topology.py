import torch

def build_tree_topology(depth=5, branch=3):
    """
    通用 k-ary 树拓扑构建器。
    返回:
        parents: List[int], 每个节点的父节点索引
        total_nodes: 树的总节点数 (不含 Root)
    """
    parents = []
    # 每一层的节点起始索引
    layer_start_indices = [0] 
    
    # 第一层 (Root 的直接子节点)
    for _ in range(branch):
        parents.append(-1)
    
    # 从第二层开始迭代
    for d in range(1, depth):
        start = len(parents) - (branch ** d)
        for i in range(branch ** d):
            parent_idx = start + i
            for _ in range(branch):
                parents.append(parent_idx)
                
    total_nodes = len(parents)
    return parents, total_nodes

def generate_tree_attention_mask(parents, seq_len, dtype=torch.float16, device="cuda"):
    """
    生成 Target Model 验证阶段所需的 Tree Attention Mask。
    包含 Prefix (下三角因果) 和 Tree (祖先可见) 两部分。
    """
    tree_nodes = len(parents)
    total_len = seq_len + tree_nodes
    
    # 初始化为负无穷 (-inf 意味着完全屏蔽)
    mask = torch.full((total_len, total_len), torch.finfo(dtype).min, device=device)
    
    # 1. 【终极修复】历史序列 (Prefix) 必须严格遵守下三角因果掩码 (Causal Mask)
    # 保证前面的词绝对看不到后面的词！
    prefix_causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=device))
    mask[:seq_len, :seq_len] = torch.where(prefix_causal_mask == 1, 0.0, torch.finfo(dtype).min)
    
    # 2. 所有的树节点 (Draft) 都可以完整看到整个历史序列 (Prefix)
    mask[seq_len:, :seq_len] = 0.0
    
    # 3. 树内部节点的可见性 (子节点只能看到路径上的祖先和自己)
    for i in range(tree_nodes):
        curr = i
        while curr != -1:
            mask[seq_len + i, seq_len + curr] = 0.0
            curr = parents[curr]
            
    # 增加 batch 和 head 维度适配 HuggingFace: [batch, 1, total_len, total_len]
    return mask.unsqueeze(0).unsqueeze(0)

def generate_position_ids(parents, base_position, device="cuda"):
    """
    根据树的深度计算绝对位置编码 (RoPE 所需)。
    同层节点共享相同的位置 ID。
    """
    tree_nodes = len(parents)
    pos_ids = torch.zeros(tree_nodes, dtype=torch.long, device=device)
    
    for i in range(tree_nodes):
        if parents[i] == -1:
            pos_ids[i] = base_position + 1
        else:
            pos_ids[i] = pos_ids[parents[i]] + 1
            
    return pos_ids.unsqueeze(0) # [1, tree_nodes]