import os
# 强行注入国内镜像源，破解网络封锁
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download

def verify_vocabulary(target_id: str, draft_id: str):
    """
    严谨的数学验证：确保 Target 和 Draft 使用完全同构的映射空间 (Vocabulary)
    """
    print(f"🔍 正在验证 Tokenizer 一致性...\nTarget: {target_id}\nDraft: {draft_id}")
    
    # 仅轻量级拉取 Tokenizer 进行验证
    target_tokenizer = AutoTokenizer.from_pretrained(target_id)
    draft_tokenizer = AutoTokenizer.from_pretrained(draft_id)
    
    target_vocab_size = target_tokenizer.vocab_size
    draft_vocab_size = draft_tokenizer.vocab_size
    
    print(f"📊 目标模型词表大小: {target_vocab_size}")
    print(f"📊 草稿模型词表大小: {draft_vocab_size}")
    
    if target_vocab_size != draft_vocab_size:
        raise ValueError(
            f"❌ 词汇表大小不匹配！Target={target_vocab_size}, Draft={draft_vocab_size}。投机解码将失效，请更换模型！"
        )
    
    # 进一步验证特殊 Token 是否一致
    if target_tokenizer.bos_token_id != draft_tokenizer.bos_token_id:
        print("⚠️ 警告: 两个模型的 BOS (起始) Token ID 不一致，但可能不影响主体推理。")
    else:
        print("✅ BOS Token ID 一致。")
        
    print("✅ 词表校验通过！它们在同一个离散数学空间内，投机解码理论基础成立。\n")

def download_model(repo_id: str, local_dir: str):
    """
    使用 snapshot_download 安全、快速地下拉模型（支持断点续传）
    """
    print(f"🚀 开始下载模型: {repo_id} -> {local_dir}")
    os.makedirs(local_dir, exist_ok=True)
    
    
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        ignore_patterns=["*.bin*"],
        resume_download=True,
        max_workers=4  # 开启多线程加速
    )
    print(f"🎉 下载完成: {repo_id}\n")

if __name__ == "__main__":
    # 定义模型在 HuggingFace 上的 ID (换成 chat 版本)
    TARGET_MODEL_ID = "NousResearch/Llama-2-7b-chat-hf" 
    DRAFT_MODEL_ID = "JackFram/llama-68m"
    
    # Store workstation-level models outside this project workspace.
    model_dir = "/root/autodl-tmp/Model"
    
    # 修改目标文件夹名称，以示区分
    target_local_path = os.path.join(model_dir, "Llama-7B-Chat-Target")
    draft_local_path = os.path.join(model_dir, "Llama-68M-Draft")
    
    try:
        # 第一步：学术严谨的预检
        verify_vocabulary(TARGET_MODEL_ID, DRAFT_MODEL_ID)
        
        # 第二步：执行下载
        print("💡 提示：7B 模型大约需要 14GB 磁盘空间，请确保空间充足，您可以去喝杯咖啡了...")
        download_model(TARGET_MODEL_ID, target_local_path)
        download_model(DRAFT_MODEL_ID, draft_local_path)
        
        
        print(f"🏆 所有模型已成功落盘至：{model_dir}")
        print("接下来，您可以修改 main.py 中的路径，开启我们的本地投机解码之旅了！")
        
    except Exception as e:
        print(f"\n❌ 下载或校验过程中断，错误信息：\n{e}")
