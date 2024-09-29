# coding=utf-8

import json

from openprompt.plms import load_plm
from openprompt.prompts import MixedTemplate, ManualVerbalizer, ManualTemplate, ProtoVerbalizer
from openprompt import PromptDataLoader
from openprompt import PromptForClassification
from datasets import load_dataset, load_from_disk
from openprompt.data_utils import InputExample
from torch.optim import AdamW
import torch
import matplotlib.pyplot as plt
from openprompt.data_utils.data_sampler import FewShotSampler
from openprompt.utils.reproduciblity import set_seed
from torch.utils.data import DataLoader

import tripe
import data_util
from adver import FGSM, PGD

# model_path = 'E://models//wobert_chinese_base'
model_path = 'E:/models/chinese-roberta-wwm-ext'

batch_size = 1
EPOCH = 20

device = 'cuda'
# learning_rate = 2e-6
learning_rate = 2e-6 # 0.847
# learning_rate = 5e-6 # 0.837
save_path = './best_val.ckpt'
alpha = 0.9

# 0.8325


# my_data_set, max_length = data_util.get_chn_data()
# my_data_set, max_length = data_util.get_hotel_data()
my_data_set, max_length = data_util.get_epr_data()

print('train dataset length: ', len(my_data_set['train']))
print('test dataset length: ', len(my_data_set['test']))


for seed in [2, 144, 145]:
    for shot in [1, 4]:

        eps = 0.5
        # 定义 PLM
        plm, tokenizer, model_config, wrapper_class = load_plm("bert", model_path)  # 本地路径
        drop_out = 0.3
        model_config.attention_probs_dropout_prob = drop_out
        model_config.hidden_dropout_prob = drop_out

        set_seed(seed)
        sampler = FewShotSampler(num_examples_per_label=shot, also_sample_dev=True, num_examples_per_label_dev=shot)
        train_dataset, valid_dataset = sampler(my_data_set['train'], seed=seed)
        # print(train_dataset)

        # print([n for n, p in plm.named_parameters()])
        prompt_template = ManualTemplate(text='{"placeholder":"text_a" }。这是{"mask"}的。', tokenizer=tokenizer)
        # prompt_template = ManualTemplate(text='{"placeholder":"text_a" }。{"meta": "key"}的评价是：{"mask"}。',
        #                                  tokenizer=tokenizer)  # 796

        print('prompt dataset length: ', len(train_dataset))
        train_dataloader = PromptDataLoader(dataset=train_dataset, template=prompt_template, tokenizer=tokenizer,
                                            tokenizer_wrapper_class=wrapper_class, max_seq_length=max_length,
                                            batch_size=batch_size)
        valid_dataloader = PromptDataLoader(dataset=valid_dataset, template=prompt_template, tokenizer=tokenizer,
                                            tokenizer_wrapper_class=wrapper_class, max_seq_length=max_length,
                                            decoder_max_length=3,
                                            batch_size=batch_size, shuffle=True, teacher_forcing=False,
                                            predict_eos_token=False,
                                            truncate_method="head")
        test_dataloader = PromptDataLoader(dataset=my_data_set['test'], template=prompt_template, tokenizer=tokenizer,
                                           tokenizer_wrapper_class=wrapper_class, max_seq_length=max_length,
                                           decoder_max_length=3,
                                           batch_size=batch_size, shuffle=True, teacher_forcing=False,
                                           predict_eos_token=False,
                                           truncate_method="head")

        classes = ['负向', '正向']

        ### chn
        # label_words = {'负向': ['坏', '差', '恶', '次', '劣', '孬', '丑'], '正向': ['好', '优', '良', '善', '妙', '美', '棒']}  # 761

        ### hotel
        # label_words = {
        #     '负向': ['坏', '差', '远', '劣', '丑', '少', '旧', '老', '贵', '简陋'],
        #     '正向': ['好', '优', '良', '善', '妙', '美', '棒', '大', '舒服', '漂亮']}

        # ### epr
        label_words = {'负向': ['坏', '差', '低', '薄', '慢', '厚', '少', '差', '恶', '次', '劣'],
                       '正向': ['好', '优', '良', '善', '妙', '美', '棒', '高', '方便']}


        prompt_verbalizer = ManualVerbalizer(classes=classes, tokenizer=tokenizer, label_words=label_words)
        prompt_model = PromptForClassification(plm=plm, template=prompt_template, verbalizer=prompt_verbalizer,
                                               freeze_plm=False).to(device)

        val_accs = []

        print(prompt_model.named_parameters())

        def evaluate(data_loader, type='validation'):
            all_preds = []
            all_labels = []
            for step, inputs in enumerate(data_loader):
                inputs.cuda()
                logits = prompt_model(inputs)
                labels = inputs['label']
                all_labels.extend(labels.cpu().tolist())
                all_preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
                # print('-' * 100)
                # print('labels is : ', labels.tolist())
                # print('prediction is : ',torch.argmax(logits, dim=-1).tolist())
                # print(logits.shape)
            acc = sum([int(i == j) for i, j in zip(all_preds, all_labels)]) / len(all_preds)
            if type == 'validation':
                val_accs.append(acc)
            print(type, ' accuracy: ', acc)
            return acc


        # train

        loss_func = torch.nn.CrossEntropyLoss()
        # tripe_loss_func = torch.nn.TripletMarginLoss(margin=3.0, reduction='sum')
        tripe_loss_func = tripe.TripletLoss()

        no_decay = ['bias', 'LayerNorm.weight']

        # optimizer_grouped_parameters1 = [
        #     {'params': [p for n, p in prompt_model.plm.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        #     {'params': [p for n, p in prompt_model.plm.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        # ]

        # Using different optimizer for prompt parameters and model parameters
        optimizer_grouped_parameters = [
            {'params': [p for n, p in prompt_model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in prompt_model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0},
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)

        ### 对比学习
        # cl_train_dataloader, cl_model = cl.get_cl_module([item.text_a for item in train_dataset], tokenizer, max_length, plm, model_config, batch_size=batch_size)
        tripe_loader, cl_model = tripe.get_loader_model(plm, model_config, train_dataset, valid_dataset, tokenizer,
                                                        max_length, len(train_dataloader))
        cl_model.train()
        prompt_model.train()
        # eps = 0.1 * shot
        # fgsm = FGSM(model=prompt_model, eps=eps)
        pgd = PGD(prompt_model)
        K = 3

        def train(EPOCH):
            # 这三个list用来绘图
            nums = []
            loss_nums = []
            loss_nums1 = []
            loss_nums2 = []
            acc_nums = []
            best_val_acc = 0
            print('-------------------start training，EPOCH = {}------------------'.format(EPOCH))
            for epoch in range(EPOCH):  # Longer epochs are needed
                tot_loss = 0
                print('tarin loader len: ', len(train_dataloader))
                print('cl tarin loader len: ', len(tripe_loader))
                for step, (inputs, tripe_data) in enumerate(zip(train_dataloader, tripe_loader)):
                    # print('step: ', step)
                    if inputs is not None:
                        inputs = inputs.to(device)
                        logits = prompt_model(inputs)
                        labels = inputs['label']
                        loss1 = loss_func(logits, labels)
                    else:
                        loss1 = 0

                    if tripe_data is not None:
                        # loss2 = 0
                        anchor_input, positive_input, negative_input = tripe_data
                        anchor_input, positive_input, negative_input = anchor_input.to(device), positive_input.to(
                            device), negative_input.to(device)
                        anchor_logits = cl_model(anchor_input['input_ids'], anchor_input['attention_mask'],
                                                 anchor_input['token_type_ids'])
                        positive_logits = cl_model(positive_input['input_ids'], positive_input['attention_mask'],
                                                   positive_input['token_type_ids'])
                        negative_logits = cl_model(negative_input['input_ids'], negative_input['attention_mask'],
                                                   negative_input['token_type_ids'])

                        loss2 = tripe_loss_func(anchor_logits, positive_logits, negative_logits)
                        # if loss2.item() == 0:
                        #     print('-' * 100)
                        #     print(anchor_logits)
                        #     print(positive_logits)
                        #     print(negative_logits)
                        #     distance_positive = (anchor_logits - positive_logits).pow(2).sum(1)  # .pow(.5)
                        #     distance_negative = (anchor_logits - negative_logits).pow(2).sum(1)
                        #     print('distance_positive, distance_negative: ',distance_positive, distance_negative)
                        # print(loss2)
                    else:
                        loss2 = 0

                    loss = alpha * loss1 + (1 - alpha) * loss2
                    # loss = loss1
                    loss.backward()
                    tot_loss += loss.item()
                    if epoch < 20:

                        # 对抗训练
                        ##########################################################################################
                        # fgsm.attack()  # 在embedding上添加对抗扰动
                        # # outputs = prompt_model(inputs)
                        # # labels = inputs['label']
                        # # loss1_adv = loss_func(outputs, labels)
                        # anchor_input, positive_input, negative_input = tripe_data
                        # anchor_input, positive_input, negative_input = anchor_input.to(device), positive_input.to(
                        #     device), negative_input.to(device)
                        # anchor_logits = cl_model(anchor_input['input_ids'], anchor_input['attention_mask'],
                        #                          anchor_input['token_type_ids'])
                        # positive_logits = cl_model(positive_input['input_ids'], positive_input['attention_mask'],
                        #                            positive_input['token_type_ids'])
                        # negative_logits = cl_model(negative_input['input_ids'], negative_input['attention_mask'],
                        #                            negative_input['token_type_ids'])
                        #
                        # loss2_adv = tripe_loss_func(anchor_logits, positive_logits, negative_logits)
                        # # loss_adv = alpha * loss1_adv + (1 - alpha) * loss2_adv
                        # loss2_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
                        # fgsm.restore()  # 恢复embedding参数
                        ##############################################################################################
                        pgd.backup_grad()
                        print('eps= ', eps)
                        for t in range(K):
                            pgd.attack(is_first_attack=(t == 0), epsilon=eps)  # 在embedding上添加对抗扰动, first attack时备份param.data
                            if t != K - 1:
                                prompt_model.zero_grad()
                                cl_model.zero_grad()
                            else:
                                pgd.restore_grad()
                            logits = prompt_model(inputs)
                            labels = inputs['label']
                            loss1_adv = loss_func(logits, labels)
                            anchor_input, positive_input, negative_input = tripe_data
                            anchor_input, positive_input, negative_input = anchor_input.to(device), positive_input.to(
                                device), negative_input.to(device)
                            anchor_logits = cl_model(anchor_input['input_ids'], anchor_input['attention_mask'],
                                                     anchor_input['token_type_ids'])
                            positive_logits = cl_model(positive_input['input_ids'], positive_input['attention_mask'],
                                                       positive_input['token_type_ids'])
                            negative_logits = cl_model(negative_input['input_ids'], negative_input['attention_mask'],
                                                       negative_input['token_type_ids'])

                            loss2_adv = tripe_loss_func(anchor_logits, positive_logits, negative_logits)
                            loss_adv = alpha * loss1_adv + (1 - alpha) * loss2_adv
                            loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
                        pgd.restore()  # 恢复embedding参数

                    optimizer.step()
                    prompt_model.zero_grad()
                    cl_model.zero_grad()
                val_acc = evaluate(valid_dataloader)
                print("Epoch {}, loss1: {}, loss2: {},total loss: {}".format(epoch, loss1.item(),
                                                                             0 if loss2 == 0 else loss2.item(),
                                                                             loss.item()), flush=True)
                nums.append(epoch)
                loss_nums.append(loss.item())
                loss_nums1.append(loss1.item())
                # loss_nums2.append(loss2.item())
                if val_acc >= best_val_acc:
                    torch.save(prompt_model.state_dict(), save_path)
                    best_val_acc = val_acc
            # 打印一下画图的参数，之后用到的时候可以直接拿来画图
            print(nums)
            print(loss_nums)

            # plt.figure()
            # plt.plot(nums, loss_nums, label='Train loss')
            # plt.plot(nums, loss_nums1, label='loss1 ')
            # plt.plot(nums, loss_nums2, label='loss2')
            # plt.title('Training loss')
            # plt.legend()
            # plt.xlabel('epoch')
            # plt.ylabel('loss')
            #
            # plt.show()


        #
        train(EPOCH)
        #
        # prompt_model.load_state_dict(torch.load(save_path))
        # prompt_model = prompt_model.cuda()
        cl_model.eval()
        prompt_model.eval()
        test_acc = evaluate(test_dataloader, 'test')
        print('test acc :', test_acc)
        with open(f'./adv_res/hotel/{eps}seed{seed}shot{shot}.txt', 'w') as f:
            f.write('seed = {}, shot = {}, acc = {}'.format(seed, shot, test_acc))

