from utils.common import _79GB, AlgoType, get_model_config


class TransformerMemory:
    def __init__(self, dtype_size, pp_size, sp_size, micro_bsz, seq_len, model_size, ckpt) -> None:
        self._dtype_size = dtype_size
        self._micro_batch_size = micro_bsz
        self._sequence_length = seq_len
        self._SP = sp_size
        self._PP = pp_size
        self.ckpt = ckpt

        self._h, self._a, self._l, self._mlp_ratio, self._multiple_of = get_model_config(model_size)

    def get_memory_threshold(self, algo: AlgoType):
        if algo == AlgoType.ISP:
            return self.get_isp_memory_threshold()
        elif algo == AlgoType.MSP:
            return self.get_msp_memory_threshold()
        elif algo == AlgoType.FSP:
            return self.get_fsp_memory_threshold()

        assert ValueError(f"unkonw algo: {algo}")

    def get_fsp_memory_threshold(self):
        activation = (
            self._dtype_size
            * self._micro_batch_size
            * self._sequence_length
            * self._h
            * (34 + (5 * self._a * self._sequence_length / (self._h * self._SP)))
            / self._SP
        ) * self._l * (1 - self.ckpt)  # 显存阈值根据pp0来计算，需要micro_num >= pp，stage_0需要保存 pp 份才成立
        memory_threshold = _79GB - activation
        return memory_threshold, activation

    def get_isp_memory_threshold(self):
        activation = (
            self._dtype_size
            * self._micro_batch_size
            * self._sequence_length
            * self._h
            * (34 + (5 * self._a * self._sequence_length / (self._h * self._SP)))
            / self._SP
        ) * self._l * (1 - self.ckpt)
        memory_threshold = _79GB - activation
        return memory_threshold, activation

    def get_msp_memory_threshold(self):
        activation = (
            self._dtype_size
            * self._micro_batch_size
            * self._sequence_length
            * self._h
            * (4 + 30 / self._SP + (5 * self._a * self._sequence_length / self._h / self._SP))
        ) * self._l * (1 - self.ckpt)
        memory_threshold = _79GB - activation
        return memory_threshold, activation
