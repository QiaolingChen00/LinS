from utils.utils import CommPredict
from utils.common import AlgoType


class TransformerCommunication:
    def __init__(self, b, s, h, num_layers, vocab_size, mlp_ratio, multiple_of, dtype_size, lins_scale=None, sp_scale=None, cost_data=None):
        self.b = b  # Batch size
        self.s = s  # Sequence length
        self.h = h  # Hidden size
        self.lins_scale = lins_scale
        self.sp_scale = sp_scale

        self.qkv_communication_latency = 0
        self.post_attention_communication_latency = 0
        self.first_linear_communication_latency = 0
        self.second_linear_communication_latency = 0
        self.attention_all_to_all_communication_latency = 0
        self.cost_data = cost_data
        
        self.mlp_ratio = mlp_ratio
        self.multiple_of = multiple_of
        self.dtype_size = dtype_size

        # self.toal_comm = self.communication_isp()

    def get_comm_cost(self, comm_alo, scale, volume):
        return self.cost_data[comm_alo].predict(scale, volume)

    def allgather(self, volume, scale):
        if scale <= 1:
            return 0
        comm_alo = "all_gahter"
        predict = self.get_comm_cost(comm_alo, scale, volume)
        # print(f"allgather:{predict}")
        return predict

    def reducescatter(self, volume, scale):
        if scale <= 1:
            return 0
        comm_alo = "reduce_scatter"
        # predict = CommPredict(volume,comm_alo,scale).prediction
        predict = self.get_comm_cost(comm_alo, scale, volume)
        # print(f"reducescatter:{predict}")
        return predict

    def alltoall(self, volume, scale):
        if scale <= 1:
            return 0
        comm_alo = "all2all"
        # predict = CommPredict(volume,comm_alo,scale).prediction
        predict = self.get_comm_cost(comm_alo, scale, volume)
        return predict

    def get_volume(self, volume, alo):
        if alo == "isp":
            return volume

        # TODO MSP,FSP etc.
        return 0

    def communication_isp(self, lins_scale, sp_scale):
        self.lins_scale = lins_scale
        self.sp_scale = sp_scale

        qkv_communication_volume = self.get_volume(3 * self.dtype_size * self.h**2, "isp")
        # forward + backward
        self.qkv_communication_latency = 2 * self.allgather(
            qkv_communication_volume, self.lins_scale
        ) + self.reducescatter(qkv_communication_volume, self.lins_scale)
        
        post_attention_communication_volume = self.get_volume(self.dtype_size * self.h**2, "isp")
        # forward + backward
        self.post_attention_communication_latency = 2 * self.allgather(
            post_attention_communication_volume, self.lins_scale
        ) + self.reducescatter(post_attention_communication_volume, self.lins_scale)
        
        mlp_hidden_size = self.multiple_of * ((int(self.h * self.mlp_ratio)+ self.multiple_of - 1) // self.multiple_of) 
        first_linear_communication_volume = self.get_volume(self.dtype_size * mlp_hidden_size * self.h, "isp")
        # forward + backward
        self.first_linear_communication_latency = 2 * self.allgather(
            first_linear_communication_volume, self.lins_scale
        ) + self.reducescatter(first_linear_communication_volume, self.lins_scale)

        second_linear_communication_volume = self.get_volume(self.dtype_size * mlp_hidden_size* self.h, "isp")
        # forward + backward
        self.second_linear_communication_latency = 2 * self.allgather(
            second_linear_communication_volume, self.lins_scale
        ) + self.reducescatter(second_linear_communication_volume, self.lins_scale)

        attention_all_to_all_communication_volume = self.get_volume(4 * self.dtype_size * self.b * self.s * self.h, "isp")
        self.attention_all_to_all_communication_latency = 2 * self.alltoall(
            attention_all_to_all_communication_volume, self.sp_scale
        )

        return (
           self.first_linear_communication_latency
            + self.second_linear_communication_latency
            + self.qkv_communication_latency
            + self.post_attention_communication_latency
        ), self.attention_all_to_all_communication_latency
    
    # TODO: xyt
    def communication_msp(self, lins_scale, sp_scale):
        pass
    
    # TODO: xyt
    def communication_fsp(self, lins_scale, sp_scale):
        pass        
    
    def communication(self, lins_scale, sp_scale, algo_type):
        if algo_type == AlgoType.ISP:
            return self.communication_isp(lins_scale, sp_scale)
        elif algo_type == AlgoType.MSP:
            return self.communication_msp(lins_scale, sp_scale)
        elif algo_type == AlgoType.FSP:
            return self.communication_fsp(lins_scale, sp_scale)
        