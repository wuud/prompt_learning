import torch
import codecs

# file1 = './Chn/chn-e1all.txt'
# file1 = './epr/chn-e1all.txt'
# file1 = './hotel/chn-e1all.txt'
# file1 = './simcse/chn-e1all.txt'
# file1 = './simcse/epr-e1all.txt'
# file1 = './simcse/hotel-e1all.txt'
# file1 = './adv_res/epr/0.3/all.txt'
file1 = './adv_res/epr/0.5/all.txt'
# file1 = './adv_res/chn/chn0.3/all.txt'


def get_data(file):
    lines = codecs.open(file)
    res = {}
    res[1] = []
    res[4] = []
    res[8] = []
    res[16] = []
    # res[32] = []
    # res[64] = []
    for line in lines:
        if len(line.strip()) == 0: continue
        segs = line.split(',')
        nums = [seg.split('=')[1].strip() for seg in segs]
        res[int(nums[1])].append(float(nums[2]) * 100)
    return res

# print(get_data(file1))
# print(get_data(file2))
res = get_data(file1)
print(res)

shot1 = torch.Tensor(res[1])
shot4 = torch.Tensor(res[4])
shot8 = torch.Tensor(res[8])
shot16 = torch.Tensor(res[16])
# shot32 = torch.Tensor(res[32])
# shot64 = torch.Tensor(res[64])
#
print('1 shot: mean={}, std = {}, max = {}'.format(torch.mean(shot1), torch.std(shot1), torch.max(shot1)))
print('4 shot: mean={}, std = {}, max = {}'.format(torch.mean(shot4), torch.std(shot4), torch.max(shot4)))
# print('8 shot: mean={}, std = {}, max = {}'.format(torch.mean(shot8), torch.std(shot8), torch.max(shot8)))
# print('16 shot: mean={}, std = {}, max = {}'.format(torch.mean(shot16), torch.std(shot16), torch.max(shot16)))
# print('32 shot: mean={}, std = {}, max = {}'.format(torch.mean(shot32), torch.std(shot32), torch.max(shot32)))
# print('64 shot: mean={}, std = {}, max = {}'.format(torch.mean(shot64), torch.std(shot64), torch.max(shot64)))
