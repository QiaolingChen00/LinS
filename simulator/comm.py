from utils import CommPredict

class TransformerCommunication:
    def __init__(self, b, s, h,num_layers,vocab_size,lins_scale,sp_scale):
        self.b = b  # Batch size
        self.s = s  # Sequence length
        self.h = h  # Hidden size
        self.lins_scale=lins_scale
        self.sp_scale=sp_scale

        self.qkv_communication_latency=0
        self.post_attention_communication_latency=0
        self.first_linear_communication_latency=0
        self.second_linear_communication_latency=0
        self.attention_all_to_all_communication_latency=0

        self.toal_comm=self.communication_isp()

    def allgather(self,volume,scale):
        comm_alo='Allgather'
        predict = CommPredict(volume,comm_alo,scale).prediction
        return predict
    
    def alltoall(self,volume,scale):
        comm_alo='Alltoall'
        predict = CommPredict(volume,comm_alo,scale).prediction
        return predict


    def get_volume(self,volume,alo):
        if alo=="isp":
            return volume
        
        # TODO MSP,FSP etc.
        return 0
    

    def communication_isp(self):
        qkv_communication_volume=self.get_volume(6*self.h**2,"isp")
        self.qkv_communication_latency=self.allgather(qkv_communication_volume,self.lins_scale)

        post_attention_communication_volume=self.get_volume(2*self.h**2,"isp")
        self.post_attention_communication_latency=self.allgather(post_attention_communication_volume,self.lins_scale)
    
        first_linear_communication_volume=self.get_volume(8*self.h**2,"isp")
        self.first_linear_communication_latency=self.allgather(first_linear_communication_volume,self.lins_scale)

        second_linear_communication_volume=self.get_volume(8*self.h**2,"isp")
        self.second_linear_communication_latency=self.allgather(second_linear_communication_volume,self.lins_scale)
       
        attention_all_to_all_communication_volume=self.get_volume(4*self.s*self.h,"isp")
        self.attention_all_to_all_communication_latency=self.alltoall(attention_all_to_all_communication_volume,self.sp_scale)

        return self.attention_all_to_all_communication_latency+self.first_linear_communication_latency+self.second_linear_communication_latency+self.qkv_communication_latency+self.post_attention_communication_latency


        
       