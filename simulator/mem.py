from utils.common import _79GB, AlgoType, get_model_config


class TransformerMemory:
    def __init__(
        self,
        dtype_size: int,
        pp_size: int,
        sp_size: int,
        micro_bsz: int,
        seq_len: int,
        model_size: int,
        ckpt: int,
        use_fa: int,
    ) -> None:
        """_summary_

        Args:
            dtype_size (int): 数据dtype大小,单位B
            pp_size (int): pipeline size
            sp_size (int): sequence size
            micro_bsz (int): micro_bsz
            seq_len (int): seq_len, 这里的seq是没有被sp切过的
            model_size (int): 模型大小,是7,13,30是以B为单位表示的参数量
            ckpt (int): 是否开启ckpt,取值为0, 1
            use_fa (int): 是否开启fa,取值为0, 1
        """
        self._dtype_size = dtype_size
        self._micro_batch_size = micro_bsz
        self._sequence_length = seq_len
        self._SP = sp_size
        self._PP = pp_size
        self.ckpt = ckpt
        self.use_fa = use_fa

        self._h, self._a, self._l, self._mlp_ratio, self._multiple_of = get_model_config(model_size)
        self._l = self._l // pp_size
        self._dtype_size = self._dtype_size // 2  # 显存计算公式是针对fp16类型数据的，所以这里先除以2

    def get_memory_threshold(self, algo: AlgoType):
        if algo == AlgoType.ISP:
            return self.get_isp_memory_threshold()
        elif algo == AlgoType.MSP:
            return self.get_msp_memory_threshold()
        elif algo == AlgoType.FSP:
            return self.get_fsp_memory_threshold()

        assert ValueError(f"unknow algo: {algo}")

    def get_isp_memory_threshold(self):
        # TODO: ht mark pp情况下，rank0的激活值会累积pp个micro_bsz，所以这里是不是还得再乘一个pp_size？
        # TODO: wgt mark 应该不需要，每个pp拿到L/pp个layer，又最多保存pp个micro_num的激活，
        # rank0相当于还是L份layer的激活
        activation = (
            (
                self._dtype_size
                * self._micro_batch_size
                * self._sequence_length
                * self._h
                * (34 + (1 - self.use_fa) * (5 * self._a * self._sequence_length / self._h))
                / self._SP
            )
            * self._l
            * (1 - self.ckpt)
        )
        memory_threshold = _79GB - activation
        return memory_threshold, activation

    def get_msp_memory_threshold(self):
        activation = (
            (
                self._dtype_size
                * self._micro_batch_size
                * self._sequence_length
                * self._h
                * (4 + 30 / self._SP + (1 - self.use_fa) * (5 * self._a * self._sequence_length / self._h / self._SP))
            )
            * self._l
            * (1 - self.ckpt)
        )
        memory_threshold = _79GB - activation
        return memory_threshold, activation

    def get_fsp_memory_threshold(self):
        activation = (
            (
                self._dtype_size
                * self._micro_batch_size
                * self._sequence_length
                * self._h
                * (34 + (1 - self.use_fa) * (5 * self._a * self._sequence_length / self._h))
                / self._SP
            )
            * self._l
            * (1 - self.ckpt)
        )  # 显存阈值根据pp0来计算，需要micro_num >= pp，stage_0需要保存 pp 份才成立
        memory_threshold = _79GB - activation
        return memory_threshold, activation
