from simulator.comm import TransformerCommunication
from simulator.comp import TransformerComputation
from utils.common import get_model_config


# 1. dtype 加入复杂度
# 2. comm 没有乘以 laynum
# 3. atten 计算还没加
# 4. mmeory check
# 5. 集成simulator
class TransformerOverlap:
    def __init__(
        self,
        micro_bsz,
        seq_len,
        vocab_size,
        dtype_size,
        model_size,
        sp_size,
        pp_size,
        world_size,
        ckpt,
        model_para,
        num_layers,
        cost_data=None,
    ):
        self.b = micro_bsz  # Batch size
        self.s = seq_len  # Sequence length
        self.vocab_size = vocab_size
        self.sp_scale = sp_size
        self.dtype_size = dtype_size
        self.world_size = world_size
        self.pp_size = pp_size

        self.h, self._a, self.num_layers, self.mlp_ratio, self.multiple_of = get_model_config(model_size)
        self.num_layers = self.num_layers // pp_size
        self.ckpt = ckpt  # the activation checkpoint
        self.model_param = model_para  # the model size

    def _get_overlap(self, lins_scale, algo_type):
        self.lins_scale = lins_scale
        # 一个transformer layer的通信时延 (forward + backward)
        comm_wp, comm_sp, comm_wdp = TransformerCommunication(
            self.b,
            self.s,
            self.h,
            self.num_layers,
            self.vocab_size,
            dtype_size=self.dtype_size,
            mlp_ratio=self.mlp_ratio,
            multiple_of=self.multiple_of,
            ckpt=self.ckpt,
            model_para=self.model_param,
        ).communication(self.lins_scale, self.sp_scale, algo_type)

        # 一个transformer layer的计算时延 (forward + backward)
        comp_wp, comp_attn = TransformerComputation(
            self.b,
            self.s,
            self.h,
            self.num_layers,
            self.vocab_size,
            dtype_size=self.dtype_size,
            mlp_ratio=self.mlp_ratio,
            multiple_of=self.multiple_of,
            sp_scale=self.sp_scale,
            ckpt=self.ckpt,
        ).total_computation(algo_type)

        return self.num_layers * (max(comm_wp, comp_wp) + comm_sp + comp_attn) + comm_wdp
