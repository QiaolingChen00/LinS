from utils.common import AlgoType


class TransformerComputation:
    def __init__(self, b, s, h, num_layers, vocab_size,sp_scale, dtype_size, mlp_ratio, multiple_of, cost_data=None, ckpt=0):
        self.b = b  # Batch size
        self.s = s  # Sequence length
        self.h = h  # Hidden size
        self.sp_scale =sp_scale
        self.qkv_computation = 0
        self.qkt_computation = 0 
        self.score_v_computation = 0
        self.post_attention_linear = 0
        self.first_linear = 0
        self.second_linear = 0
        self.logits_computation = 0
        self.attention_computation = 0
        self.flash_attention_computation=0
        self.mlp_computation = 0
        self.cost_data = cost_data
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.dtype_size = dtype_size
        self.mlp_ratio = mlp_ratio
        self.multiple_of = multiple_of
        self.mlp_hidden_size = self.multiple_of * ((int(self.h * self.mlp_ratio)+ self.multiple_of - 1) // self.multiple_of) 
        self.ckpt = ckpt
        # self.comp = self.total_computation(num_layers, vocab_size)

    def get_linear_cost(self, complexity):
        return self.cost_data["linear"].predict(1, complexity)
    
    def _compute_embedding(self, scale):
        '''
        the head computation is the same as embedding computation.
        msp and fsp share the same computation.
        
        scale: the scale factor. when the algo is isp, the scale is one; else the scale is self.sp_scale.
        '''
        volumn = self.dtype_size * self.b * self.s * self.vocab_size * self.h / scale
        latency = self.get_linear_cost(volumn)
        return latency
    
    def _compute_linears(self):
        '''
        compute the latency for linears in one transformer layer, such as wqkv, wo, mlp
        '''
        
        # wqkv
        # ISP: (b, s/sp, h) * (h, 3h)
        # MSP or FSP: (b, s, h) * (h, 3h/sp)
        qkv_volumn = 3 * self.dtype_size * self.b * self.s * self.h * self.h / self.sp_scale
        qkv_latency = self.get_linear_cost(qkv_volumn)
        
        # wo
        # ISP: (b, s/sp, h) * (h, h)
        # MSP or FSP: (b, s, h/sp) * (h/sp, h)
        wo_volumn = self.dtype_size * self.b * self.s * self.h * self.h / self.sp_scale 
        wo_latency = self.get_linear_cost(wo_volumn)
        
        # mlp w1
        # ISP: (b, s/sp, h) * (h, mlp_h)
        # MSP or FSP: (b, s, h) * (h, mlp_h/sp)
        w1_volumn = self.dtype_size * self.s * self.h * self.mlp_hidden_size / self.sp_scale
        w1_latency = self.get_linear_cost(w1_volumn)
        
        # mlp w2
        # ISP: (b, s/sp, h) * (h, mlp_h)
        # MSP or FSP: (b, s, h/sp) * (h/sp, mlp_h)
        w2_volumn = self.dtype_size * self.s * self.h * self.mlp_hidden_size / self.sp_scale
        w2_latency = self.get_linear_cost(w2_volumn)
        
        
        total_latency = qkv_latency + wo_latency + w1_latency + w2_latency
        
        return total_latency
    
    def _compute_attn(self):
        '''
        compute the latency for attention in one transformer layer
        '''
        # QK^T matrix multiplication
        # (b, s, h/sp) * (b, s, h/sp)^T
        qkt_volume = self.dtype_size * self.s * self.s * self.h / self.sp_scale
        qkt_latency = self.get_linear_cost(qkt_volume)
        
        # Score dot V
        # (b, s, s) * (b, s, h/sp)
        score_v_volume = self.dtype_size * self.s * self.s * self.h / self.sp_scale
        score_v_latency = self.get_linear_cost(score_v_volume)
        
        total_latency = qkt_latency + score_v_latency
        
        return total_latency

    def _computation(self, embedding_scale):
        # TODO: the following computation exclude norm computation
        '''
        ckpt: activation checkpoint {0 or 1}
        
        the computation latency for each transformer layer
        
        compu(msp) = compu(forward) + compu(backward)
        
        compu(backward) = 2 * compu(forward)
        
        compu(forward) = (compu(linear, (wqkv, wo, mlp)) + compu(attn)) * (ckpt + 1)
        '''
        
        # compute the latency for embedding and head
        embedding_latency = self._compute_embedding(embedding_scale)
        head_latency = embedding_latency
        
        # compute the latency for linears
        linears_latency = self._compute_linears() * (self.ckpt + 1)
        
        # compute the latency for attention
        attn_latency = self._compute_attn() * (self.ckpt + 1)
        
        # the computation for each transformer layer
        # transformer_latency = linears_latency + attn_latency
        
        return linears_latency, attn_latency

    def total_computation(self, algo_type):
        
        if algo_type == AlgoType.ISP:
            # return self.total_computation_isp()
            return self._computation(1.0)
        else:
            return self._computation(self.sp_scale)
        


# Example usage
# Assuming values for b (batch size), s (sequence length), h (hidden size), num_layers, and vocab_size
# b, s, h, num_layers, vocab_size = 1, 16384, 4096, 32, 10000
# transformer_comp = TransformerComputation(b, s, h,num_layers,vocab_size)
