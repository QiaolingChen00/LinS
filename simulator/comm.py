from utils.common import AlgoType, CostType, SovlerType
from utils.utils import CommPredict

from simulator.ab_cost_model import get_comm_cost


class TransformerCommunication:
    def __init__(self, b, s, h, num_layers, vocab_size, mlp_ratio, multiple_of, dtype_size, lins_scale=None, sp_scale=None, cost_data=None, ckpt=0, model_para=0, wdp_size=1):
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
        self.mlp_hidden_size = self.multiple_of * ((int(self.h * self.mlp_ratio)+ self.multiple_of - 1) // self.multiple_of) 
        
        self.ckpt = ckpt # activation checkpoint
        self.model_para = model_para
        self.wdp_size = wdp_size

        # self.toal_comm = self.communication_isp()

    def allgather(self, volume, scale):
        if scale <= 1:
            return 0
        predict = get_comm_cost(SovlerType.MODEL, CostType.ALLGATHER, scale, volume)
        return predict

    def reducescatter(self, volume, scale):
        if scale <= 1:
            return 0
        predict = get_comm_cost(SovlerType.MODEL, CostType.REDUCESCATTER, scale, volume)
        return predict

    def alltoall(self, volume, scale):
        if scale <= 1:
            return 0
        predict = get_comm_cost(SovlerType.MODEL, CostType.ALL2ALL, scale, volume)
        return predict
    
    def allreduce(self, volume, scale):
        if scale <= 1:
            return 0
        predict = get_comm_cost(SovlerType.MODEL, CostType.ALLREDUCE, scale, volume)
        return predict

    def communication_isp(self, lins_scale, sp_scale):
        '''
        ckpt: means the activation checkpoint, {0 or 1}
        
        sp communication:
        
        comm(sp) = comm(forward, sp) + comm(backward, sp)
        
        comm(forward, sp) = 4 * comm(all2all, s/sp, b, h/sp) * (ckpt + 1)

        comm(backward, sp) = 4 * comm(all2all, s/sp, b, h/sp)
        
        wp communication: (In our implementation, the wp communication of ckpt==1 is the same as ckpt==0)
        
        comm(wp) = comm(forwad, wp) + comm(backward, wp)
        
        comm(forward, wp) = comm(all_gather, (wqkv, wo, mlp))
        
        comm(backward, wp) = comm(all_gather, (wqkv, wo, mlp)) + comm(reduceScatter, (wqkv, wo, mlp))  
        
        wdp communication: (actually wdp communication should be included in the optimizer communication)
        '''
        
        self.lins_scale = lins_scale
        self.sp_scale = sp_scale
        
        # wp communication
        qkv_wp_volume = 3 * self.dtype_size * self.h ** 2
        wo_wp_volume = self.dtype_size * self.h ** 2
        mlp_w1_volume = self.dtype_size * self.h * self.mlp_hidden_size
        mlp_w2_volume = mlp_w1_volume
        
        qkv_latency = 2 * self.allgather(qkv_wp_volume, self.lins_scale) + self.reducescatter(qkv_wp_volume, self.lins_scale)
        wo_latency = 2 * self.allgather(wo_wp_volume, self.lins_scale) + self.reducescatter(wo_wp_volume, self.lins_scale)
        mlp_w1_latency = 2 * self.allgather(mlp_w1_volume, self.lins_scale) + self.reducescatter(mlp_w1_volume, self.lins_scale)
        mlp_w2_latency = mlp_w1_latency
        
        # sp communication
        all2all_volume = self.s / self.sp_scale * self.b * self.h * self.dtype_size 
        all2all_latency = self.alltoall(all2all_volume, self.sp_scale)
        
        wp_comm_latency = qkv_latency + wo_latency + mlp_w1_latency + mlp_w2_latency
        sp_comm_latency = 4 * all2all_latency * (self.ckpt + 1) + 4 * all2all_latency # forward + backward
        
        # wdp communication
        wdp_volume = self.model_para / lins_scale
        wdp_latency = self.allreduce(wdp_volume, self.wdp_size)
        
        return wp_comm_latency, sp_comm_latency, wdp_latency
    
    def communication_msp(self, lins_scale, sp_scale):
        '''
        ckpt: means the activation checkpoint, {0 or 1}
        
        sp communication:
        
        comm(sp) = comm(forward, sp) + comm(backward, sp)
        
        comm(forward, sp) = (2 * comm(all_gather, s/sp, b, h) + 2 * comm(reduceScatter, s, b, h) + 4 * comm(all2all, s/sp, b, h/sp)) * (ckpt + 1)

        comm(backward, sp) = 2 * comm(reduceScatter, s, b, h) + 2 * comm(all_gather, s/sp, b, h) + 4 * comm(all2all, s/sp, b, h/sp)
        
        wp communication:
        
        comm(wp) = comm(forwad, wp) + comm(backward, wp)
        
        comm(forward, wp) = comm(all_gather, (wqkv, wo, mlp))
        
        comm(backward, wp) = comm(all_gather, (wqkv, wo, mlp)) + comm(reduceScatter, (wqkv, wo, mlp))  
        
        wdp communication: (actually wdp communication should be included in the optimizer communication)
        '''
        self.lins_scale = lins_scale
        self.sp_scale = sp_scale

        # compute sp communication
        # all_gather and reduceScatter have the same commu volume
        # the communication volume in backward is equal to the forward
        qkv_sp_volume = self.s * self.b * self.h * self.dtype_size # the forward all-gather
        wo_sp_volume = self.s * self.b * self.h * self.dtype_size # the forward reduceScatter
        mlp_w1_sp_volume = qkv_sp_volume # the forward all-gather
        mlp_w2_sp_volume = self.s * self.b * self.h * self.dtype_size # the forward reduceScatter
        
        # all2all
        all2all_sp_volume = self.s / self.sp_scale * self.b * self.h * self.dtype_size
        
        # compute the sp latency (forward + backward)
        sp_forward = self.allgather(qkv_sp_volume, self.sp_scale) + self.reducescatter(wo_sp_volume, self.sp_scale) \
                     + self.allgather(mlp_w1_sp_volume, self.sp_scale) + self.reducescatter(mlp_w2_sp_volume, self.sp_scale) \
                     + 4 * self.alltoall(all2all_sp_volume, self.lins_scale)
        
        sp_backward = self.reducescatter(qkv_sp_volume, self.sp_scale) + self.allgather(wo_sp_volume, self.sp_scale) \
                      + self.reducescatter(mlp_w1_sp_volume, self.sp_scale) + self.allgather(mlp_w2_sp_volume, self.sp_scale) \
                      + 4 * self.alltoall(all2all_sp_volume, self.lins_scale)
                      
        sp_forward = sp_forward * (self.ckpt + 1)
        
        sp_comm_latency = sp_forward + sp_backward
        
        # commpute wp communication
        qkv_wp_volume = 3 * self.h * self.h / self.sp_scale * self.dtype_size
        wo_wp_volume = self.h * self.h / self.sp_scale * self.dtype_size
        
        # w2 and w3 have the same volume as w1
        mlp_w1_wp_volume = self.h * self.mlp_hidden_size / self.sp_scale * self.dtype_size
        
        qkv_wp_latency = 2 * self.allgather(qkv_wp_volume, self.lins_scale) + self.reducescatter(qkv_wp_volume, self.lins_scale)
        wo_wp_latency = 2 * self.allgather(wo_wp_volume, self.lins_scale) + self.reducescatter(wo_wp_volume, self.lins_scale)
        mlp_w1_wp_latency = 2 * self.allgather(mlp_w1_wp_volume, self.lins_scale) + self.reducescatter(mlp_w1_wp_volume, self.lins_scale)
        mlp_w2_wp_latency = mlp_w1_wp_latency
        
        wp_comm_latency = qkv_wp_latency + wo_wp_latency + mlp_w1_wp_latency + mlp_w2_wp_latency
        
        # wdp communication
        wdp_volume = self.model_para // self.sp_scale // self.lins_scale
        wdp_latency = self.allreduce(wdp_volume, self.wdp_size)
        
        return wp_comm_latency, sp_comm_latency, wdp_latency
    
    def communication_fsp(self, lins_scale, sp_scale):
        '''
        ckpt: means the activation checkpoint, {0 or 1}
        
        sp communication:
        
        comm(sp) = comm(forward, sp) + comm(backward, sp)
        
        comm(forward, sp) = (2 * comm(all_gather, s/sp, b, h) + 2 * comm(reduceScatter, s, b, h) + 4 * comm(all2all, s/sp, b, h/sp)) * (ckpt + 1)

        comm(backward, sp) = 2 * comm(reduceScatter, s, b, h) + 4 * comm(all_gather, s/sp, b, h) + 4 * comm(all2all, s/sp, b, h/sp)
        
        wp communication:
        
        comm(wp) = comm(forwad, wp) + comm(backward, wp)
        
        comm(forward, wp) = comm(all_gather, (wqkv, wo, mlp))
        
        comm(backward, wp) = comm(all_gather, (wqkv, wo, mlp)) + comm(reduceScatter, (wqkv, wo, mlp))  
        
        wdp communication: (actually wdp communication should be included in the optimizer communication)
        '''
        self.lins_scale = lins_scale
        self.sp_scale = sp_scale

        # compute sp communication
        # all_gather and reduceScatter have the same commu volume
        # the communication volume in backward is equal to the forward
        qkv_sp_volume = self.s * self.b * self.h * self.dtype_size # the forward all-gather
        wo_sp_volume = self.s * self.b * self.h * self.dtype_size # the forward reduceScatter
        mlp_w1_sp_volume = qkv_sp_volume # the forward all-gather
        mlp_w2_sp_volume = self.s * self.b * self.h * self.dtype_size # the forward reduceScatter
        
        # all2all
        all2all_sp_volume = self.s / self.sp_scale * self.b * self.h * self.dtype_size 
        
        # compute the sp latency (forward + backward)
        sp_forward = self.allgather(qkv_sp_volume, self.sp_scale) + self.reducescatter(wo_sp_volume, self.sp_scale) \
                     + self.allgather(mlp_w1_sp_volume, self.sp_scale) + self.reducescatter(mlp_w2_sp_volume, self.sp_scale) \
                     + 4 * self.alltoall(all2all_sp_volume, self.lins_scale)
        
        sp_backward = self.allgather(qkv_sp_volume, self.sp_scale) + self.reducescatter(qkv_sp_volume, self.sp_scale) \
                      + self.allgather(wo_sp_volume, self.sp_scale) \
                      + self.allgather(mlp_w1_sp_volume, self.sp_scale) + self.reducescatter(mlp_w1_sp_volume, self.sp_scale) \
                      + self.allgather(mlp_w2_sp_volume, self.sp_scale) + 4 * self.alltoall(all2all_sp_volume, self.lins_scale) 
        
        sp_forward = sp_forward * (self.ckpt + 1)

        sp_comm_latency = sp_forward + sp_backward
        
        # commpute wp communication
        qkv_wp_volume = 3 * self.h * self.h / self.sp_scale * self.dtype_size
        wo_wp_volume = self.h * self.h / self.sp_scale * self.dtype_size
        
        # w2 and w3 have the same volume as w1
        mlp_w1_wp_volume = self.h * self.mlp_hidden_size / self.sp_scale * self.dtype_size
        
        qkv_wp_latency = 2 * self.allgather(qkv_wp_volume, self.lins_scale) + self.reducescatter(qkv_wp_volume, self.lins_scale)
        wo_wp_latency = 2 * self.allgather(wo_wp_volume, self.lins_scale) + self.reducescatter(wo_wp_volume, self.lins_scale)
        mlp_w1_wp_latency = 2 * self.allgather(mlp_w1_wp_volume, self.lins_scale) + self.reducescatter(mlp_w1_wp_volume, self.lins_scale)
        mlp_w2_wp_latency = mlp_w1_wp_latency
        
        wp_comm_latency = qkv_wp_latency + wo_wp_latency + mlp_w1_wp_latency + mlp_w2_wp_latency
        
        # wdp communication
        wdp_volume = self.model_para // self.sp_scale // self.lins_scale
        wdp_latency = self.allreduce(wdp_volume, self.wdp_size)
        
        return wp_comm_latency, sp_comm_latency, wdp_latency
    
    
    def communication(self, lins_scale, sp_scale, algo_type):
        if algo_type == AlgoType.ISP:
            return self.communication_isp(lins_scale, sp_scale)
        elif algo_type == AlgoType.MSP:
            return self.communication_msp(lins_scale, sp_scale)
        elif algo_type == AlgoType.FSP:
            return self.communication_fsp(lins_scale, sp_scale)
        raise ValueError(f"Unkoen algo_type: {algo_type}")
        