"""
main package
"""
from simulator.simulator import Simulator


def main():
    """main function"""
    config = {
        "world_size": 1024,
        "global_batch_size": 4096,
        "sequence_length": 1024,
        "model_size":30,
        "grad_acc":1,
    }

    for sp in [2,4,8,16]:
        for world_size in [128,256,512,1024]:
            for model_size in [7,13,30,65]:
                print(f'\n SP={sp},world_size{world_size},model_size{model_size}')
                config.update({"SP":sp})
                config.update({"world_size":world_size})
                config.update({"world_size":model_size})

                simulator = Simulator(config)
                simulator.run()


if __name__ == "__main__":
    main()
