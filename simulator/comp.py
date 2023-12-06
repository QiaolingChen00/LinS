class TransformerComputation:
    def __init__(self, b, s, h, num_layers, vocab_size, cost_data=None):
        self.b = b  # Batch size
        self.s = s  # Sequence length
        self.h = h  # Hidden size
        self.qkv_computation = 0
        self.qkt_computation = 0
        self.score_v_computation = 0
        self.post_attention_linear = 0
        self.first_linear = 0
        self.second_linear = 0
        self.logits_computation = 0
        self.attention_computation = 0
        self.mlp_computation = 0
        self.cost_data = cost_data
        self.dtype_c = 1  # bfloat16
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        # self.comp = self.total_computation(num_layers, vocab_size)

    def get_linear_cost(self, complexity):
        return self.cost_data["linear"].predict(1, complexity)

    def compute_attention_block(self):
        # Calculate Q, K, V
        # self.qkv_computation = 6 * self.b * self.s * self.h**2
        self.qkv_computation_one_linear = self.b * self.s * self.h**2
        self.qkv_computation_lat = self.dtype_c * 3 * self.get_linear_cost(self.qkv_computation_one_linear)

        # QK^T matrix multiplication
        # self.qkt_computation = 2 * self.b * self.s**2 * self.h
        self.qkt_computation_one_linear = self.b * self.s**2 * self.h
        # TODO, make dtype_c as input key
        self.qkt_computation_lat = self.dtype_c * self.get_linear_cost(self.qkt_computation_one_linear)

        # Score dot V
        self.score_v_computation = self.qkt_computation  # Same as self.qkt_computation
        self.score_v_computation_lat = self.get_linear_cost(self.score_v_computation)

        # Linear mapping after attention
        # self.post_attention_linear = 2 * self.b * self.s * self.h**2
        self.post_attention_linear_one_linear = self.b * self.s * self.h**2
        self.post_attention_linear_lat = self.dtype_c * self.get_linear_cost(self.post_attention_linear_one_linear)

        # Total computation for attention block
        # total_attention_computation = self.qkv_computation + self.qkt_computation + self.score_v_computation + self.post_attention_linear
        total_attention_computation_lat = (
            self.qkv_computation_lat
            + self.qkt_computation_lat
            + self.score_v_computation_lat
            + self.post_attention_linear_lat
        )
        return total_attention_computation_lat

    def compute_mlp_block(self):
        # First linear layer
        self.first_linear = 4 * self.b * self.s * self.h**2
        self.first_linear_lat = self.dtype_c * self.get_linear_cost(self.first_linear)

        # Second linear layer
        self.second_linear = self.first_linear  # Same as self.first_linear

        # Total computation for MLP block
        # total_mlp_computation = self.first_linear + self.second_linear
        return self.first_linear_lat * 2

    def compute_logits(self):
        # Logits computation
        # TODO: fix me!
        self.logits_computation = 2 * self.b * self.s * self.h * self.vocab_size
        self.logits_computation_lat = self.get_linear_cost(self.logits_computation)
        return self.logits_computation_lat

    def total_computation(self):
        # Compute total for each block
        self.attention_computation = self.compute_attention_block()
        self.mlp_computation = self.compute_mlp_block()
        self.logits_computation = self.compute_logits()

        # Total computation for one transformer layer
        per_layer_computation = self.attention_computation + self.mlp_computation

        # Total computation for all layers
        total_computation = self.num_layers * per_layer_computation + self.logits_computation
        return total_computation


# Example usage
# Assuming values for b (batch size), s (sequence length), h (hidden size), num_layers, and vocab_size
# b, s, h, num_layers, vocab_size = 1, 16384, 4096, 32, 10000
# transformer_comp = TransformerComputation(b, s, h,num_layers,vocab_size)
