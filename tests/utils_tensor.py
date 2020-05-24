import torch
from utils import duplicate_tensor_to_cpu, dynamic_cat_sent

a = torch.randn([10, 3])
b = duplicate_tensor_to_cpu(a, torch.int64)
print(b.shape)

a = torch.LongTensor().new_full([1, 1], 0)
b = torch.LongTensor().new_full([5, 10], 233)
c = dynamic_cat_sent(a, b, 0)

