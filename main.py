"""
main package
"""
from simulator.simulator import Simulator


def _get_bs(global_bs, world_size, sequence_length):
    num_tokens = global_bs * 1024
    bs_bn = num_tokens // world_size // sequence_length

    factors = []
    for B in range(1, int(bs_bn**0.5) + 1):
        if bs_bn % B == 0:
            C = bs_bn // B
            factors.append((B, C))
    return factors


def main():
    """main function"""
    config = {
        "world_size": 64,
        "global_batch_size": 4096,
        "sequence_length": 4096,
        "model_size": 7,
        "grad_acc": 1,
        "SP": 1,
        "micro_bs": 1,
        "grad_acc": 1,
        "vocab_size": 103168,
    }

    cost_data_path = "/mnt/petrelfs/wangguoteng.p/ds_comm_bench/LinS/data/cost_data.pickle"
    # world_size = 64
    # for sp in [2,4,8,16]:
    #     for model_size in [7]:
    #         print(f'\n SP={sp},world_size{world_size},model_size{model_size}')
    #         config.update({"SP":sp})
    simulator = Simulator(config, cost_data_path=cost_data_path)
    simulator.run()

    # bs_bn=_get_bs(config['global_batch_size'],config['world_size'],config['sequence_length'])
    # print(bs_bn)
    # for i in range(len(bs_bn)):
    #     micro_bs,grad_acc=bs_bn[i][0],bs_bn[i][1]
    #     config.update({"micro_bs":micro_bs})
    #     config.update({"grad_acc":grad_acc})
    #     print(f'\n\nmicro_bs{micro_bs}')
    #     for sp in [1,8,16]:
    #         config.update({"SP":sp})
    #         print(f'\nSP={sp}')



if __name__ == "__main__":
    main()
