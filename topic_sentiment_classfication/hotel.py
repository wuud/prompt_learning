# coding=utf-8
import codecs

from datasets import load_dataset, load_from_disk
from openprompt.data_utils import InputExample, FewShotSampler
from openprompt.plms import load_plm
from openprompt.prompts import ManualVerbalizer, KnowledgeableVerbalizer, ManualTemplate
import torch
from openprompt.utils.reproduciblity import set_seed
import json
from torch.optim import AdamW
from openprompt import PromptDataLoader
from openprompt import PromptForClassification
import matplotlib.pyplot as plt

model_path = 'E://models//chinese-roberta-wwm-ext'
train_data_path = "./dataset/hotel_train.txt"
test_data_path = "./dataset/hotel_test.txt"
batch_size = 8

# learning_rate = 1e-6, seed = 1, shot = 24, 0.7774769897130482
max_length = 128
EPOCH = 15
device = 'cuda'
shot = 16
learning_rate = 1e-6
seed = 1
set_seed(seed)

my_data_set = {}


def get_data_set(file, split):
    my_data_set[split] = []
    lines = codecs.open(file, encoding='utf-8').readlines()
    for idx, line in enumerate(lines):
        eles = line.split(',')
        # print(eles[0], eles[1], int(eles[3]))
        input = InputExample(guid=idx, label= 0 if int(eles[2]) == -1 else 1, text_a=eles[0], meta={'key': eles[1]})
        my_data_set[split].append(input)
        # print(input)


get_data_set(train_data_path, 'train')
get_data_set(test_data_path, 'test')
print('train dataset length: ', len(my_data_set['train']))
# print('train dataset length: ', my_data_set['train'][0:5])
print('test dataset length: ', len(my_data_set['test']))


for seed in [2, 3, 143, 144, 145]:
    for shot in [1, 4, 8, 16]:

        sampler = FewShotSampler(num_examples_per_label=shot, also_sample_dev=True, num_examples_per_label_dev=shot)
        train_dataset, valid_dataset = sampler(my_data_set['train'], seed=seed)


        plm, tokenizer, model_config, WrapperClass = load_plm("bert", model_path)


        classes = ['负向', '正向']
        # label_words.txt = ['负向', '正向'] # 755

        # label_words.txt = {'负向': ['一般', '差', '小', '旧', '很小', '老', '少'], '正向': ['好', '不错', '方便', '干净', '大', '很大', '高']}
        # label_words = {
        #     '负向': ['坏', '差', '远', '劣', '丑', '少', '旧', '老', '贵'],
        #     '正向': ['好', '优', '良', '善', '妙', '美', '棒', '大', '舒服', '漂亮']}

        label_words = {
            '负向': ['坏', '差', '远', '劣', '丑', '少', '旧', '老', '贵', '简陋'],
            '正向': ['好', '优', '良', '善', '妙', '美', '棒', '大', '舒服', '漂亮']}

        my_verbalizer = ManualVerbalizer(tokenizer=tokenizer, classes=classes, label_words=label_words)

        # prompt_template = ManualTemplate(text='{"placeholder":"text_a" }。这是{"mask"}的。', tokenizer=tokenizer) # 763
        # prompt_template = ManualTemplate(text='{"placeholder":"text_a" }。对于{"meta": "key"}这个主题，评价是{"mask"}的。', tokenizer=tokenizer) # 781
        # prompt_template = ManualTemplate(text='{"placeholder":"text_a" }。对于{"meta": "key"}，评价是{"mask"}。', tokenizer=tokenizer) # 787
        prompt_template = ManualTemplate(text='{"placeholder":"text_a" }。{"meta": "key"}的评价是：{"mask"}。', tokenizer=tokenizer) # 796
        # prompt_template = ManualTemplate(text='文本：{"placeholder":"text_a" }。在上述文本中{"meta": "key"}的评价是：{"mask"}。', tokenizer=tokenizer) # 792
        #

        use_cuda = True

        prompt_model = PromptForClassification(plm=plm,template=prompt_template, verbalizer=my_verbalizer, freeze_plm=False).to(device)

        train_dataloader = PromptDataLoader(dataset=train_dataset, template=prompt_template, tokenizer=tokenizer,
                                            tokenizer_wrapper_class=WrapperClass, max_seq_length=max_length)
        valid_dataloader = PromptDataLoader(dataset=valid_dataset, template=prompt_template, tokenizer=tokenizer,
                                            tokenizer_wrapper_class=WrapperClass, max_seq_length=max_length,
                                            decoder_max_length=3,
                                            batch_size=batch_size, shuffle=True, teacher_forcing=False, predict_eos_token=False,
                                            truncate_method="head")
        test_dataloader = PromptDataLoader(dataset=my_data_set['test'], template=prompt_template, tokenizer=tokenizer,
                                           tokenizer_wrapper_class=WrapperClass, max_seq_length=max_length,
                                           decoder_max_length=3,
                                           batch_size=batch_size, shuffle=True, teacher_forcing=False, predict_eos_token=False,
                                           truncate_method="head")


        val_accs = []


        def evaluate(model, data_loader, type='validation'):
            all_preds = []
            all_labels = []
            for step, inputs in enumerate(data_loader):
                inputs.cuda()
                logits = model(inputs)
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
            print(type, ' accuracy:', acc)
            return acc

        # train

        loss_func = torch.nn.CrossEntropyLoss()

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in prompt_model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in prompt_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)

        def train(EPOCH):
            # 这三个list用来绘图
            nums = []
            loss_nums = []
            acc_nums = []
            best_val_acc  = 0
            print('-------------------start training，EPOCH = {}------------------'.format(EPOCH))
            for epoch in range(EPOCH):  # Longer epochs are needed
                tot_loss = 0
                for step, inputs in enumerate(train_dataloader):
                    inputs = inputs.to(device)
                    # print('-' * 100)
                    # print(inputs['input_ids'][0])
                    # print(tokenizer.decode(inputs['input_ids'][0]))
                    logits = prompt_model(inputs)
                    labels = inputs['label']
                    loss = loss_func(logits, labels)
                    loss.backward()
                    tot_loss += loss.item()
                    # torch.nn.utils.clip_grad_norm_(prompt_model.template.parameters(), 1.0)
                    # optimizer1.step()
                    # optimizer1.zero_grad()
                    optimizer.step()
                    optimizer.zero_grad()
                val_acc = evaluate(prompt_model, valid_dataloader)
                print("Epoch {}, average loss: {}".format(epoch, tot_loss / (step + 1)), flush=True)
                nums.append(epoch)
                loss_nums.append(tot_loss / (step + 1))
                # if val_acc >= best_val_acc:
                #     torch.save(prompt_model.state_dict(), "./best_val.ckpt")
                #     best_val_acc = val_acc

            # 打印一下画图的参数，之后用到的时候可以直接拿来画图
            print(nums)
            print(loss_nums)
            # plt.figure()
            # plt.plot(nums, loss_nums, label='Train loss')
            # plt.plot(nums, val_accs, label='Validation acc')
            # plt.title('Training loss')
            # plt.legend()
            # plt.xlabel('epoch')
            # plt.ylabel('loss')
            #
            # plt.show()


        #
        # train(EPOCH)
        # prompt_model.load_state_dict(torch.load("./best_val.ckpt"))
        # prompt_model = prompt_model.cuda()

        # torch.cuda.empty_cache()

        acc = evaluate(prompt_model, test_dataloader, 'test')
        # with open(f'./res/hotel/seed{seed}_shot{shot}.txt', 'w') as f:
        #     f.write('seed = {}, shot = {}, acc = {}'.format(seed, shot, acc))
