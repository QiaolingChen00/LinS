import pickle

from simulator.comm import TransformerCommunication
from simulator.comp import TransformerComputation


# 1. dtype 加入复杂度
# 2. comm 没有乘以 laynum
# 3. atten 计算还没加
# 4. mmeory check
# 5. 集成simulator
class TransformerOverlap:
    def __init__(self, b, s, h, num_layers, vocab_size, dtype_size, mlp_ratio, multiple_of, lins_scale=None, sp_scale=None, cost_data=None):
        self.b = b  # Batch size
        self.s = s  # Sequence length
        self.h = h  # Hidden size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.lins_scale = lins_scale
        self.sp_scale = sp_scale
        self.dtype_size = dtype_size
        self.mlp_ratio = mlp_ratio
        self.multiple_of = multiple_of

        self.cost_data = cost_data
        assert cost_data is not None
        # self.overlap = self._get_overlap()

    def _get_overlap(self, lins_scale, sp_scale):
        self.lins_scale = lins_scale
        self.sp_scale = sp_scale
        # 一个transformer layer的通信时延
        comm_wp,comm_sp = TransformerCommunication(
            self.b, self.s, self.h, self.num_layers, self.vocab_size, dtype_size=self.dtype_size, mlp_ratio=self.mlp_ratio, multiple_of=self.multiple_of, cost_data=self.cost_data
        ).communication_isp(self.lins_scale, self.sp_scale)
        
        # 一个transformer layer的计算时延
        comp_wp,comp_attn = TransformerComputation(
            self.b, self.s, self.h, self.num_layers, self.vocab_size, dtype_size=self.dtype_size, mlp_ratio=self.mlp_ratio, multiple_of=self.multiple_of, cost_data=self.cost_data,sp_scale=self.sp_scale
        ).total_computation()
        # print(f"comm:{comm}, comp:{comp}")
        # return comm - comp if comm > comp else 0
        # print(f"comm_wp:{comm_wp},comm_sp:{comm_sp},comp_wp:{comp_wp}, comp_attn:{comp_attn}")
        return max(comm_wp, comp_wp)+comm_sp+comp_attn


def main(args=None):
    cost_data_path = "/mnt/petrelfs/wangguoteng.p/ds_comm_bench/LinS/data/cost_data.pickle"
    with open(cost_data_path, "r") as f:
        cost_data = pickle.load(cost_data_path)

    overlap_res = TransformerOverlap(1, 4096, 4096, 32, 10000, 64, 8, cost_data=cost_data)
    # print(overlap_res._get_overlap())



if __name__ == "__main__":
    main()
