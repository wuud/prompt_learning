# coding=utf-8

from openprompt.plms import load_plm
from openprompt.prompts import MixedTemplate, ManualVerbalizer, ManualTemplate, AutomaticVerbalizer
from openprompt import PromptDataLoader
from openprompt import PromptForClassification
from datasets import load_dataset, load_from_disk
from openprompt.data_utils import InputExample
from torch.optim import AdamW
import torch
import matplotlib.pyplot as plt
from openprompt.data_utils.data_sampler import FewShotSampler
from openprompt.utils.reproduciblity import set_seed
import random
import utils as my_util

from tqdm import tqdm

torch.autograd.set_detect_anomaly(True)
# model_path = 'E://models//wobert_chinese_base'
model_path = 'E://models//chinese-roberta-wwm-ext'
batch_size = 4
EPOCH = 10
device = 'cuda'

# shot = 16
# learning_rate = 5e-5  # csldcp
learning_rate = 1e-6    # chn
# learning_rate = 5e-6
# learning_rate = 3e-5
# 0.5621890547263682
seed = 144

for shot in [1, 4,8,16]:
    # dataset, max_length, classes = my_util.get_cnews_data()
    # dataset, max_length, classes = my_util.get_tnews_data()
    # dataset, max_length, classes = my_util.get_csldcp_data()
    # dataset, max_length, classes = my_util.get_chn_data()
    # dataset, max_length, classes = my_util.get_epr_data()
    dataset, max_length, classes = my_util.get_hotel_data()

    set_seed(seed)
    # 定义 PLM
    plm, tokenizer, model_config, wrapper_class = load_plm("bert", model_path)  # 本地路径
    # plm, tokenizer, model_config, wrapper_class = load_plm("bert", 'bert-base-chinese') # huggingface 仓库

    sampler = FewShotSampler(num_examples_per_label=shot, also_sample_dev=True, num_examples_per_label_dev=shot)
    train_dataset, valid_dataset = sampler(dataset['train'])
    # print(train_dataset)
    # template

    # prompt_template = ManualTemplate(text='{"placeholder":"text_a" }。好好思考一下，这包括{"mask"}。', tokenizer=tokenizer)  # template1
    # prompt_template = ManualTemplate(text='{"placeholder":"text_a" }。这包括{"mask"}。', tokenizer=tokenizer)  # template1
    # prompt_template = ManualTemplate(text='{"placeholder":"text_a" }。这是{"mask"}的。', tokenizer=tokenizer)
    prompt_template = ManualTemplate(text='{"placeholder":"text_a" }。{"meta": "key"}的评价是：{"mask"}。',
                                     tokenizer=tokenizer)

    train_dataloader = PromptDataLoader(dataset=train_dataset, template=prompt_template, tokenizer=tokenizer,
                                        tokenizer_wrapper_class=wrapper_class, max_seq_length=max_length)
    valid_dataloader = PromptDataLoader(dataset=valid_dataset, template=prompt_template, tokenizer=tokenizer,
                                        tokenizer_wrapper_class=wrapper_class, max_seq_length=max_length,
                                        decoder_max_length=3,
                                        batch_size=batch_size, shuffle=True, teacher_forcing=False,
                                        predict_eos_token=False,
                                        truncate_method="head")
    test_dataloader = PromptDataLoader(dataset=dataset['test'], template=prompt_template, tokenizer=tokenizer,
                                       tokenizer_wrapper_class=wrapper_class, max_seq_length=max_length,
                                       decoder_max_length=3,
                                       batch_size=batch_size, shuffle=True, teacher_forcing=False,
                                       predict_eos_token=False,
                                       truncate_method="head")

    prompt_verbalizer = AutomaticVerbalizer(classes=classes, tokenizer=tokenizer)
    # prompt_verbalizer = ManualVerbalizer(classes=classes, tokenizer=tokenizer).from_file('verbalizer.txt')
    # prompt_verbalizer = ManualVerbalizer(classes=classes, tokenizer=tokenizer, label_words.txt=label_words.txt)
    prompt_model = PromptForClassification(plm=plm, template=prompt_template, verbalizer=prompt_verbalizer,
                                           freeze_plm=False).to(device)

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


    def prompt_initialize(verbalizer, prompt_model, init_dataloader):
        dataloader = init_dataloader
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Init_using_{}".format("train")):
                batch = batch.cuda()
                logits = prompt_model(batch)
            verbalizer.optimize_to_initialize()


    # train

    loss_func = torch.nn.CrossEntropyLoss()

    prompt_initialize(prompt_verbalizer, prompt_model, train_dataloader)

    no_decay = ['bias', 'LayerNorm.weight']

    # it's always good practice to set no decay to biase and LayerNorm parameters
    optimizer_grouped_parameters1 = [
        {'params': [p for n, p in prompt_model.plm.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in prompt_model.plm.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]

    # Using different optimizer for prompt parameters and model parameters

    optimizer1 = AdamW(optimizer_grouped_parameters1, lr=learning_rate)

    # tot_step = len(train_dataloader) // args.gradient_accumulation_steps * args.max_epochs
    # scheduler1 = get_linear_schedule_with_warmup(
    #     optimizer1,
    #     num_warmup_steps=0, num_training_steps=tot_step)

    optimizer2 = None
    scheduler2 = None


    # print('++' * 100)
    # print(prompt_model.verbalizer.parameters())

    def train(EPOCH):
        # 这三个list用来绘图
        nums = []
        loss_nums = []
        acc_nums = []
        best_val_acc = 0
        print('-------------------start training，EPOCH = {}------------------'.format(EPOCH))
        for epoch in range(EPOCH):  # Longer epochs are needed
            tot_loss = 0
            for step, inputs in enumerate(train_dataloader):
                inputs = inputs.to(device)
                logits = prompt_model(inputs)
                labels = inputs['label']
                loss = loss_func(logits, labels)
                loss.backward()
                tot_loss += loss.item()
                # torch.nn.utils.clip_grad_norm_(prompt_model.template.parameters(), 1.0)
                # optimizer1.step()
                # optimizer1.zero_grad()
                optimizer1.step()
                optimizer1.zero_grad()
            val_acc = evaluate(prompt_model, valid_dataloader)
            print("Epoch {}, average loss: {}".format(epoch, tot_loss / (step + 1)), flush=True)
            nums.append(epoch)
            loss_nums.append(tot_loss / (step + 1))
            if val_acc >= best_val_acc:
                torch.save(prompt_model.state_dict(), "./best_val.ckpt")
                best_val_acc = val_acc

        # 打印一下画图的参数，之后用到的时候可以直接拿来画图
        print(nums)
        print(loss_nums)
        #
        # plt.figure()
        # plt.plot(nums, loss_nums, label='Train loss')
        # plt.plot(nums, val_accs, label='Validation acc')
        # plt.title('Training loss')
        # plt.legend()
        # plt.xlabel('epoch')
        # plt.ylabel('loss')
        #
        # plt.show()


    train(EPOCH)
    # prompt_model.load_state_dict(torch.load("./best_val.ckpt"))
    prompt_model = prompt_model.cuda()

    torch.cuda.empty_cache()

    acc = evaluate(prompt_model, test_dataloader, 'test')
    with open(f'./res/hotel/shot{shot}.txt', 'w') as f:
        f.write('seed = {}, shot = {}, acc = {}'.format(seed, shot, acc))
