import torch

####################################### tnews ##############################################

# seed = 2, 1: 5427860696517413, 4: 5502487562189055, 8：5671641791044776, 16: 5781094527363184 ,
# seed = 144, 1: 5577114427860697, 4: 5502487562189055, 8:572139303482587, 16: 5691542288557214 ,
# seed = 145, 1: 56318407960199, 4: 5527363184079602,8: 5601990049751244, 16: 5810945273631841 ,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   64: 5691542288557214
shot1 = torch.Tensor([55.77114427860697, 54.27860696517413, 56.318407960199])
shot4 = torch.Tensor([56.06965174129354, 55.02487562189055, 55.27363184079602])
shot8 = torch.Tensor([56.71641791044776, 57.2139303482587, 56.01990049751244])
shot16 = torch.Tensor([57.81094527363184, 56.91542288557214, 58.10945273631841])

print('1 shot: mean={}, std = {}'.format(torch.mean(shot1), torch.std(shot1)))
print('4 shot: mean={}, std = {}'.format(torch.mean(shot4), torch.std(shot4)))
print('8 shot: mean={}, std = {}'.format(torch.mean(shot8), torch.std(shot8)))
print('16 shot: mean={}, std = {}'.format(torch.mean(shot16), torch.std(shot16)))

####################################### cnews ##############################################
# seed = 2, shot = 1, acc = 7963
# seed = 2, shot = 4, acc = 8455
# seed = 2, shot = 8, acc = 9284
# seed = 2, shot = 16, acc = 9279
# seed = 144, shot = 1, acc = 8127
# seed = 144, shot = 4, acc = 9246
# seed = 144, shot = 8, acc = 9312
# seed = 144, shot = 16, acc = 946
# seed = 143, 1: 7808, 4: 8371, 8: 8524, 16: 9529

shot1 = torch.Tensor([79.63, 81.27, 78.08])
shot4 = torch.Tensor([84.55, 92.46, 83.71])
shot8 = torch.Tensor([92.84, 93.12, 85.24])
shot16 = torch.Tensor([92.79, 94.6, 95.29])

print('1 shot: mean={}, std = {}'.format(torch.mean(shot1), torch.std(shot1)))
print('4 shot: mean={}, std = {}'.format(torch.mean(shot4), torch.std(shot4)))
print('8 shot: mean={}, std = {}'.format(torch.mean(shot8), torch.std(shot8)))
print('16 shot: mean={}, std = {}'.format(torch.mean(shot16), torch.std(shot16)))
#######################################csldcp##############################################

# seed = 2: 1: 4798206278026906, 4: 5056053811659192, 8:  5263452914798207, 16: 5571748878923767
# seed = 144: 1: 48710762331838564, 4: 5196188340807175, 8:  5347533632286996, 16: 5599775784753364
# seed = 145: 1: 48654708520179374, 4: 5123318385650224, 8:  5386771300448431, 16: 5644618834080718

shot1 = torch.Tensor([47.98206278026906, 48.710762331838564, 48.654708520179374])
shot4 = torch.Tensor([50.56053811659192, 51.96188340807175, 51.23318385650224])
shot8 = torch.Tensor([52.63452914798207,  53.47533632286996, 53.86771300448431])
shot16 = torch.Tensor([55.71748878923767, 55.99775784753364, 56.44618834080718])

print('1 shot: mean={}, std = {}'.format(torch.mean(shot1), torch.std(shot1)))
print('4 shot: mean={}, std = {}'.format(torch.mean(shot4), torch.std(shot4)))
print('8 shot: mean={}, std = {}'.format(torch.mean(shot8), torch.std(shot8)))
print('16 shot: mean={}, std = {}'.format(torch.mean(shot16), torch.std(shot16)))
