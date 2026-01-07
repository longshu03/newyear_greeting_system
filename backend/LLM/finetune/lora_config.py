from peft import LoraConfig,TaskType


def get_lora_config() -> LoraConfig:
    """
    创建并返回LoRA（Low-Rank Adaptation）配置对象。

    LoRA是一种参数高效的微调方法，它通过向模型中的线性层添加低秩分解矩阵来适应新任务，
    从而大大减少需要训练的参数数量。

    Returns:
        LoraConfig: LoRA配置对象，包含以下参数：
        - r (int): LoRA的秩，决定低秩矩阵的大小。秩越小，参数越少，但能力可能受限。
        - lora_alpha (int): LoRA缩放因子，用于调整低秩矩阵对原始权重的影响。
        - target_modules (List[str]): 要应用LoRA的模块名称列表。这里指定了"q_proj"和"v_proj"，
          即自注意力机制中的查询和值投影层。
        - lora_dropout (float): LoRA层的dropout率，用于防止过拟合。
        - task_type (str): 任务类型，这里设置为"CAUSAL_LM"表示因果语言建模（生成任务）。
    
    示例:
        >>> lora_config = get_lora_config() 
        >>> print(lora_config)
        LoraConfig(r=8, lora_alpha=32, ...)
    """
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,  # LoRA的秩，通常设置为8、16等较小值
        lora_alpha=32,  # 缩放因子，通常设置为秩的两倍或更高
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # 在查询和值投影层应用LoRA
        lora_dropout=0.1,  # 10%的dropout率
        task_type="CAUSAL_LM",  # 任务类型为因果语言建模
        bias="none"
    )