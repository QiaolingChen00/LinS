class TransformerComputation:
    def __init__(self, b, s, h, num_layers, vocab_size,sp_scale, dtype_size, mlp_ratio, multiple_of, cost_data=None):
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
        # self.comp = self.total_computation(num_layers, vocab_size)

    def get_linear_cost(self, complexity):
        return self.cost_data["linear"].predict(1, complexity)

    def compute_attention_block(self):
        # Calculate Q, K, V
        # self.qkv_computation = 6 * self.b * self.s * self.h**2
        self.qkv_computation_one_linear = self.dtype_size * 3 * self.b * self.s/self.sp_scale * self.h**2
        self.qkv_computation_lat = self.get_linear_cost(self.qkv_computation_one_linear)

        # QK^T matrix multiplication
        # self.qkt_computation = 2 * self.b * self.s**2 * self.h
        self.qkt_computation_one_linear = self.dtype_size * self.b * self.s**2 * self.h/self.sp_scale
        self.qkt_computation_lat = self.get_linear_cost(self.qkt_computation_one_linear)

        # Score dot V
        self.score_v_computation = self.qkt_computation  # Same as self.qkt_computation
        self.score_v_computation_lat = self.get_linear_cost(self.score_v_computation)

        # Linear mapping after attention
        # self.post_attention_linear = 2 * self.b * self.s * self.h**2
        self.post_attention_linear_one_linear = self.dtype_size * self.b * self.s/self.sp_scale * self.h**2
        self.post_attention_linear_lat = self.get_linear_cost(self.post_attention_linear_one_linear)

        # Total computation for attention block
        # total_attention_computation = self.qkv_computation + self.qkt_computation + self.score_v_computation + self.post_attention_linear
        total_attention_computation_lat = (
            self.qkv_computation_lat
            + self.post_attention_linear_lat
        )

        total_flash_attentino_computation_lat=(
             self.qkt_computation_lat
            + self.score_v_computation_lat)
        return total_attention_computation_lat, total_flash_attentino_computation_lat

    def compute_mlp_block(self):
        # First linear layer
        mlp_hidden_size = self.multiple_of * ((int(self.h * self.mlp_ratio)+ self.multiple_of - 1) // self.multiple_of) 
        self.first_linear = self.dtype_size  * self.b * self.s * mlp_hidden_size * self.h/self.sp_scale
        self.first_linear_lat = self.get_linear_cost(self.first_linear)

        # Second linear layer
        self.second_linear = self.first_linear  # Same as self.first_linear

        # Total computation for MLP block
        # total_mlp_computation = self.first_linear + self.second_linear
        return self.first_linear_lat * 2

    def compute_logits(self):
        # Logits computation
        self.logits_computation = self.dtype_size * self.b * self.s * self.h * self.vocab_size
        self.logits_computation_lat = self.get_linear_cost(self.logits_computation)
        return self.logits_computation_lat

    def total_computation(self):
        # Compute total for each block
        
        # xyt：attention_compuattion: wqkv and wo
        # xyt: flash_attention_computation: actual attention computation
        self.attention_computation,self.flash_attention_computation = self.compute_attention_block()
        
        
        self.mlp_computation = self.compute_mlp_block()
        
        self.logits_computation = self.compute_logits()

        # Total computation for one transformer layer
        per_layer_computation = self.attention_computation + self.mlp_computation
        

        # Total computation for all layers
        total_computation = self.num_layers * per_layer_computation + 2 * self.logits_computation
        total_flash_computation=self.num_layers * self.flash_attention_computation
        
        # 返回的是forward+backward；其中backward=forward*2
        return 3 * total_computation, 3 * total_flash_computation


# Example usage
# Assuming values for b (batch size), s (sequence length), h (hidden size), num_layers, and vocab_size
# b, s, h, num_layers, vocab_size = 1, 16384, 4096, 32, 10000
# transformer_comp = TransformerComputation(b, s, h,num_layers,vocab_size)
