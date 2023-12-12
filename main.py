"""
main package
"""
from simulator.simulator import Simulator


def _get_bs(global_bs, world_size, sequence_length):
    num_tokens = global_bs * 1024
    # TODO：支持更多的切分策略
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
        "world_size": 128,
        "global_batch_size": 4096,
        "sequence_length": 16384,
        "model_size": 65,
        "grad_acc": 1,
        "SP": 4,
        "micro_bs": 1,
        "grad_acc": 1,
        "vocab_size": 103168,

    }

    cost_data_path = "data/cost_data.pickle"
    # world_size = 64
    # for sp in [2,4,8,16]:
    #     config.update({"SP":sp})
    #     print(f'\nSP={sp}')
    #     print(config)


    bs_bn=_get_bs(config['global_batch_size'],config['world_size'],config['sequence_length'])
    print(bs_bn)
    for i in range(len(bs_bn)):
        micro_bs,grad_acc=bs_bn[i][0],bs_bn[i][1]
        config.update({"micro_bs":micro_bs})
        config.update({"grad_acc":grad_acc})
        print(f'\n\nmicro_bs{micro_bs}')
        for sp in [1,2,4,8,16]:
            config.update({"SP":sp})
            print(f'\nSP={sp}')
            simulator = Simulator(config, cost_data_path=cost_data_path)
            simulator.run()




if __name__ == "__main__":
    main()
