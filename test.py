from simulator.comp import TransformerComputation
from simulator.comm import TransformerCommunication
import pickle

class TransformerOverlap:
    def __init__(self, b, s, h,num_layers,vocab_size,lins_scale,sp_scale, cost_data=None):
        self.b = b  # Batch size
        self.s = s  # Sequence length
        self.h = h  # Hidden size
        self.num_layers=num_layers
        self.vocab_size=vocab_size
        self.lins_scale=lins_scale
        self.sp_scale=sp_scale

        self.cost_data = cost_data
        assert cost_data is not None
        self.overlap=self._get_overlap()

    def _get_overlap(self):
        comm=TransformerCommunication(self.b,self.s,self.h,self.num_layers,self.vocab_size,self.lins_scale,self.sp_scale, cost_data=self.cost_data).toal_comm
        comp=TransformerComputation(self.b,self.s,self.h,self.num_layers,self.vocab_size, cost_data=self.cost_data).comp
        print(f'comm:{comm}, comp:{comp}')
        return comm[0] - comp if comm[0] > comp else 0

def main(args=None):
    cost_data_path = "/mnt/petrelfs/wangguoteng.p/ds_comm_bench/LinS/data/cost_data.pickle"
    with open(cost_data_path, 'rb') as f:
        cost_data = pickle.load(f)

    overlap_res=TransformerOverlap(1,4096,4096,32,10000,64,8, cost_data=cost_data)
    print(overlap_res.overlap)


if __name__ == "__main__":
    main()