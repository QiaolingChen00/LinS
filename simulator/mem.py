from utils.common import AlgoType


# 所有公式计算都变成无状态的,计算结果完全由外部传入的参数决定，内部不进行诸如切pp这样的操作
def get_isp_memory_threshold(
    dtype_size: int,
    micro_batch_size: int,
    sequence_length: int,
    hidden_dim: int,
    use_fa: int,
    head_num: int,
    layer_num: int,
    activation_ckpt: int,
    sp_size: int,
):
    """
    Args:
        dtype_size (int): bf16=2, fp32=4
        micro_batch_size (int):
        sequence_length (int):
        hidden_dim (int):
        use_fa (int): 0 or 1
        head_num (int):
        layer_num (int):
        activation_ckpt (int): 0 or 1

    Returns:
        int: activation memory usage.
    """
    # TODO: ht mark pp情况下，rank0的激活值会累积pp个micro_bsz，所以这里是不是还得再乘一个pp_size？
    # TODO: wgt mark 应该不需要，每个pp拿到L/pp个layer，又最多保存pp个micro_num的激活，
    # rank0相当于还是L份layer的激活
    if activation_ckpt:
        layer_num = 1

    activation = (
        dtype_size
        * micro_batch_size
        * sequence_length
        * hidden_dim
        * (34 + (1 - use_fa) * (5 * head_num * sequence_length / hidden_dim))
        / sp_size
    ) * layer_num
    return activation


def get_msp_memory_threshold(
    dtype_size: int,
    micro_batch_size: int,
    sequence_length: int,
    hidden_dim: int,
    use_fa: int,
    head_num: int,
    layer_num: int,
    activation_ckpt: int,
    sp_size: int,
):
    if activation_ckpt:
        layer_num = 1

    activation = (
        dtype_size
        * micro_batch_size
        * sequence_length
        * hidden_dim
        * (4 + 30 / sp_size + (1 - use_fa) * (5 * head_num * sequence_length / hidden_dim / sp_size))
    ) * layer_num
    return activation


def get_fsp_memory_threshold(
    dtype_size: int,
    micro_batch_size: int,
    sequence_length: int,
    hidden_dim: int,
    use_fa: int,
    head_num: int,
    layer_num: int,
    activation_ckpt: int,
    sp_size: int,
):
    if activation_ckpt:
        layer_num = 1

    activation = (
        dtype_size
        * micro_batch_size
        * sequence_length
        * hidden_dim
        * (34 + (1 - use_fa) * (5 * head_num * sequence_length / hidden_dim))
        / sp_size
    ) * layer_num  # 显存阈值根据pp0来计算，需要micro_num >= pp，stage_0需要保存 pp 份才成立
    return activation


def get_memory_threshold(
    algo: AlgoType,
    **kwargs,
):
    """get_memory_threshold 获得一层激活的显存占用
    注意:
    (1) seqlen一定是没有被sp切过的
    (2) 公式是基于fp16计算的, 所以传入的 dtype_size 要除以2
    Args:
        dtype_size (int): 数据元素大小, 单位B
        seq_len (int): 没有被切过的seq_len

    Returns:
        float : 一个layer的显存占用, 单位B
    """
    if algo == AlgoType.ISP:
        return get_isp_memory_threshold(**kwargs)
    elif algo == AlgoType.MSP:
        return get_msp_memory_threshold(**kwargs)
    elif algo == AlgoType.FSP:
        return get_fsp_memory_threshold(**kwargs)

    assert ValueError(f"unknow algo: {algo}")
