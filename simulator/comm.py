class TransformerCommunication:
    def __init__(self, b, s, h,num_layers,vocab_size):
        self.b = b  # Batch size
        self.s = s  # Sequence length
        self.h = h  # Hidden size

        self.qkv_communication_latency=0
        self.post_attention_communication_latency=0
        self.first_linear_communication_latency=0
        self.second_linear_communication_latency=0
        self.attention_all_to_all_communication_latency=0

        self.communication_isp()

    def allgather(self,volume):
        return volume


    def get_volume(self,volume,alo):
        if alo=="isp":
            return volume
        return 0
    

    def communication_isp(self):
        qkv_communication_volume=self.get_volume(6*self.h**2,"isp")
        self.qkv_communication_latency=self.allgather(qkv_communication_volume)

        post_attention_communication_volume=self.get_volume(2*self.h**2,"isp")
        self.post_attention_communication_latency=self.allgather(post_attention_communication_volume)
    
        first_linear_communication_volume=self.get_volume(8*self.h**2,"isp")
        self.first_linear_communication_latency=self.allgather(first_linear_communication_volume)

        second_linear_communication_volume=self.get_volume(8*self.h**2,"isp")
        self.second_linear_communication_latency=self.allgather(second_linear_communication_volume)
       