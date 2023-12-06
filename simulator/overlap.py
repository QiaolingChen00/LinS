from comp import TransformerComputation
from comm import TransformerCommunication

class TransformerOverlap:
    def __init__(self, b, s, h,num_layers,vocab_size,lins_scale,sp_scale):
        self.b = b  # Batch size
        self.s = s  # Sequence length
        self.h = h  # Hidden size
        self.num_layers=num_layers
        self.vocab_size=vocab_size
        self.lins_scale=lins_scale
        self.sp_scale=sp_scale

        self.overlap=self._get_overlap()

    def _get_overlap(self):
        comm=TransformerCommunication(self.b,self.s,self.h,self.num_layers,self.vocab_size,self.lins_scale,self.sp_scale).toal_comm
        comp=TransformerComputation(self.b,self.s,self.h,self.num_layers,self.vocab_size).comp
        print(f'comm:{comm},comp:{comp}')
        return comm[0] - comp if comm[0] > comp else 0

def main(args=None):
    overlap_res=TransformerOverlap(1,32768,5120,40,10000,64,8)
    print(overlap_res)


if __name__ == "__main__":
    main()