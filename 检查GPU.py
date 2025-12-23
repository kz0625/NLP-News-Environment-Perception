import torch
print(torch.cuda.current_device())  # 检查当前设备索引
print(torch.cuda.get_device_name(0))  # 检查设备名称
