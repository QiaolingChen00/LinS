class TransformerComputation:
    def __init__(self, b, s, h,num_layers,vocab_size):
        self.b = b  # Batch size
        self.s = s  # Sequence length
        self.h = h  # Hidden size
        self.qkv_computation=0
        self.qkt_computation =0
        self.score_v_computation=0
        self.post_attention_linear=0
        self.first_linear=0
        self.second_linear=0
        self.logits_computation=0
        self.attention_computation=0
        self.mlp_computation=0
        self.comp=self.total_computation(num_layers,vocab_size)


    def compute_attention_block(self):
        # Calculate Q, K, V
        self.qkv_computation = 6 * self.b * self.s * self.h**2

        # QK^T matrix multiplication
        self.qkt_computation = 2 * self.b * self.s**2 * self.h

        # Score dot V
        self.score_v_computation = self.qkt_computation  # Same as self.qkt_computation

        # Linear mapping after attention
        self.post_attention_linear = 2 * self.b * self.s * self.h**2

        # Total computation for attention block
        total_attention_computation = self.qkv_computation + self.qkt_computation + self.score_v_computation + self.post_attention_linear
        return total_attention_computation

    def compute_mlp_block(self):
        # First linear layer
        self.first_linear = 8 * self.b * self.s * self.h**2

        # Second linear layer
        self.second_linear = self.first_linear  # Same as self.first_linear

        # Total computation for MLP block
        total_mlp_computation = self.first_linear + self.second_linear
        return total_mlp_computation

    def compute_logits(self, vocab_size):
        # Logits computation
        self.logits_computation = 2 * self.b * self.s * self.h * vocab_size
        return self.logits_computation

    def total_computation(self, num_layers, vocab_size):
        # Compute total for each block
        self.attention_computation = self.compute_attention_block()
        self.mlp_computation = self.compute_mlp_block()
        self.logits_computation = self.compute_logits(vocab_size)

        # Total computation for one transformer layer
        per_layer_computation = self.attention_computation + self.mlp_computation

        # Total computation for all layers
        total_computation = num_layers * per_layer_computation + self.logits_computation
        print(total_computation)
        return total_computation

# Example usage
# Assuming values for b (batch size), s (sequence length), h (hidden size), num_layers, and vocab_size
b, s, h, num_layers, vocab_size = 1, 16384, 4096, 32, 10000
transformer_comp = TransformerComputation(b, s, h,num_layers,vocab_size)

