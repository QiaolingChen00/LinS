from simulator.predict_cost_model import CostModel
from utils.common import CostType, get_global_rank, get_world_size
from utils.config import Config

# if __name__ == "__main__":
#     total_results = {
#         '32' : {
#             '1': [{'lat' : 0.01}],
#             '2': [{'lat' : 0.02}],
#             '3': [{'lat' : 0.03}],
#         },
#         '16' : {
#             '1': [{'lat' : 0.01}],
#             '2': [{'lat' : 0.02}],
#             '3': [{'lat' : 0.03}],
#         }
#     }
#     print(reformat_data_to_cost_model(total_results))


if __name__ == "__main__":
    world_size = get_global_rank()
    if world_size > 1:
        build_process_gourp(world_size)

    build_type_list = [CostType.LINEAR, CostType.FLASH_ATTN]
    cost_model = CostModel(is_master=get_global_rank() == 0, re_build_cost_data=True, build_type_list=build_type_list)
