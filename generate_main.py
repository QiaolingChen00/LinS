import subprocess

# 读取main_template.py文件内容
template_file_path = "main_template.py"
with open(template_file_path, "r") as template_file:
    template_content = template_file.read()

# 配置参数
# model_size_list = [7, 13, 30, 65]
# seq_len_list = [4096, 8192, 16384, 32768, 65536, 131072, 262144]
model_size_list = [65]
seq_len_list = [131072]

# 生成配置文件并执行命令
for model_size in model_size_list:
    for seq_len in seq_len_list:
        # 根据model_size选择world_size的值
        world_size = 128

        # 替换模板内容中的字段
        config_content = (
            template_content.replace("{model_size}", str(model_size))
            .replace("{seq_len}", str(seq_len))
            .replace("{world_size}", str(world_size))
        )

        # 生成配置文件路径
        config_file_path = f"./main_{model_size}_{seq_len}.py"

        # 写入配置文件
        with open(config_file_path, "w") as config_file:
            config_file.write(config_content)

        # 执行命令并提取最后一行结果
        command = f"python {config_file_path} 2>&1 | tee ./outputs/output_{model_size}_{seq_len}.log"
        result = subprocess.run(command, shell=True, executable="/bin/bash", capture_output=True, text=True)
        output_lines = result.stdout.splitlines()
        last_line = output_lines[-1]

        # 将生成的output文件提取最后一行的配置结果，并写入final_outputs目录下，文件名保持不变
        with open(f"./sim_configs/output_{model_size}_{seq_len}.py", "w") as final_output_file:
            final_output_file.write(last_line)

print("配置文件生成和执行完成。")
