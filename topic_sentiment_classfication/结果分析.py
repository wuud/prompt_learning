import torch
import codecs


# file = './FT/res/hotel/res_all.txt'
file = './res/hotel/all.txt'
lines = codecs.open(file)
res = {}
res[1] = []
res[4] = []
res[8] = []
res[16] = []
for line in lines:
    segs = line.split(',')
    nums = [seg.split('=')[1].strip() for seg in segs]
    res[int(nums[1])].append(float(nums[2]) * 100)

print(res)

shot1 = torch.Tensor(res[1])
shot4 = torch.Tensor(res[4])
shot8 = torch.Tensor(res[8])
shot16 = torch.Tensor(res[16])

print('1 shot: mean={}, std = {}, max = {}'.format(torch.mean(shot1), torch.std(shot1), torch.max(shot1)))
print('4 shot: mean={}, std = {}, max = {}'.format(torch.mean(shot4), torch.std(shot4), torch.max(shot4)))
print('8 shot: mean={}, std = {}, max = {}'.format(torch.mean(shot8), torch.std(shot8), torch.max(shot8)))
print('16 shot: mean={}, std = {}, max = {}'.format(torch.mean(shot16), torch.std(shot16), torch.max(shot16)))
