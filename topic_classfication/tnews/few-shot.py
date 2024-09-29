# few-shot demo，要观察小样本训练的结果和不训练的结果
from tqdm import tqdm
from openprompt.data_utils.text_classification_dataset import AgnewsProcessor, DBpediaProcessor, ImdbProcessor, \
    AmazonProcessor
from openprompt.data_utils.huggingface_dataset import YahooAnswersTopicsProcessor
import torch
from openprompt.data_utils.utils import InputExample
import argparse
import numpy as np
from openprompt import PromptDataLoader
from openprompt.prompts import ManualVerbalizer, KnowledgeableVerbalizer, SoftVerbalizer, AutomaticVerbalizer
from openprompt.prompts import ManualTemplate
from openprompt.utils.calibrate import calibrate
# parser = argparse.ArgumentParser("")
#
# parser.add_argument("--plm_eval_mode", action="store_true")
# parser.add_argument("--verbalizer", type=str)
# parser.add_argument("--calibration", action="store_true")
#
# parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

model = 'roberta'
model_name_or_path = 'E:\\models\\roberta-base'  # base版本跑起来更快
result_file = './res.txt'
shot = 5
seed = 144
template_id = 0
template_id = 0
filter = 'tfidf_filter'
dataset = 'imdb'
epochs = 5
kptw_lr = 0.06
pred_temp = 1.0
max_token_split = -1
data_path = 'E:\workspace\python\KnowledgeablePromptTuning-main'
verbalizer_name = 'kpt'

import random

this_run_unicode = str(random.randint(0, 1e10))

from openprompt.utils.reproduciblity import set_seed

set_seed(seed)

from openprompt.plms import load_plm

plm, tokenizer, model_config, WrapperClass = load_plm(model, model_name_or_path)

dataset = {}

dataset['train'] = ImdbProcessor().get_train_examples(f"{data_path}/imdb/")
dataset['test'] = ImdbProcessor().get_test_examples(f"{data_path}/imdb/")
class_labels = ImdbProcessor().get_labels()
scriptsbase = "TextClassification/imdb"
scriptformat = "txt"
cutoff = 0
max_seq_l = 512
batch_s = 1
calibration = True

mytemplate = ManualTemplate(tokenizer=tokenizer).from_file(f"{data_path}/imdb/manual_template.txt", choice=template_id)

myverbalizer = KnowledgeableVerbalizer(tokenizer, classes=class_labels, candidate_frac=cutoff, pred_temp=pred_temp,
                                       max_token_split=max_token_split).from_file(
    f"{data_path}/imdb/knowledgeable_verbalizer.txt")

# (contextual) calibration

# if calibration or filter != "none":
from openprompt.data_utils.data_sampler import FewShotSampler

support_sampler = FewShotSampler(num_examples_total=200, also_sample_dev=False)
dataset['support'] = support_sampler(dataset['train'], seed=seed)

# for example in dataset['support']:
#     example.label = -1 # remove the labels of support set for clarification
support_dataloader = PromptDataLoader(dataset=dataset["support"], template=mytemplate, tokenizer=tokenizer,
                                      tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l,
                                      decoder_max_length=3,
                                      batch_size=batch_s, shuffle=False, teacher_forcing=False,
                                      predict_eos_token=False,
                                      truncate_method="tail")

from openprompt import PromptForClassification

use_cuda = True
prompt_model = PromptForClassification(plm=plm, template=mytemplate, verbalizer=myverbalizer, freeze_plm=False)
if use_cuda:
    prompt_model = prompt_model.cuda()


if calibration or filter != "none":
    org_label_words_num = [len(prompt_model.verbalizer.label_words[i]) for i in range(len(class_labels))]
    # calculate the calibration logits
    cc_logits = calibrate(prompt_model, support_dataloader)
    print("the calibration logits is", cc_logits)
    print("origial label words num {}".format(org_label_words_num))

if calibration:
    myverbalizer.register_calibrate_logits(cc_logits)
    new_label_words_num = [len(myverbalizer.label_words[i]) for i in range(len(class_labels))]
    print("After filtering, number of label words per class: {}".format(new_label_words_num))


    # register the logits to the verbalizer so that the verbalizer will divide the calibration probability in producing label logits
    # currently, only ManualVerbalizer and KnowledgeableVerbalizer support calibration.

from openprompt.data_utils.data_sampler import FewShotSampler

sampler = FewShotSampler(num_examples_per_label=shot, also_sample_dev=True, num_examples_per_label_dev=shot)
dataset['train'], dataset['validation'] = sampler(dataset['train'], seed=seed)

train_dataloader = PromptDataLoader(dataset=dataset["train"], template=mytemplate, tokenizer=tokenizer,
                                    tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l,
                                    decoder_max_length=3,
                                    batch_size=batch_s, shuffle=True, teacher_forcing=False, predict_eos_token=False,
                                    truncate_method="tail")

validation_dataloader = PromptDataLoader(dataset=dataset["validation"], template=mytemplate, tokenizer=tokenizer,
                                         tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l,
                                         decoder_max_length=3,
                                         batch_size=batch_s, shuffle=False, teacher_forcing=False,
                                         predict_eos_token=False,
                                         truncate_method="tail")

# zero-shot test
test_dataloader = PromptDataLoader(dataset=dataset["test"], template=mytemplate, tokenizer=tokenizer,
                                   tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l, decoder_max_length=3,
                                   batch_size=batch_s, shuffle=False, teacher_forcing=False, predict_eos_token=False,
                                   truncate_method="tail")


def evaluate(prompt_model, dataloader, desc):
    prompt_model.eval()
    allpreds = []
    alllabels = []
    pbar = tqdm(dataloader, desc=desc)
    for step, inputs in enumerate(pbar):
        if use_cuda:
            inputs = inputs.cuda()
        logits = prompt_model(inputs)
        labels = inputs['label']
        alllabels.extend(labels.cpu().tolist())
        allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
    acc = sum([int(i == j) for i, j in zip(allpreds, alllabels)]) / len(allpreds)
    return acc


############
#############
###############

from transformers import AdamW, get_linear_schedule_with_warmup

loss_func = torch.nn.CrossEntropyLoss()


def prompt_initialize(verbalizer, prompt_model, init_dataloader):
    dataloader = init_dataloader
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Init_using_{}".format("train")):
            batch = batch.cuda()
            logits = prompt_model(batch)
        verbalizer.optimize_to_initialize()


no_decay = ['bias', 'LayerNorm.weight']

# it's always good practice to set no decay to biase and LayerNorm parameters
optimizer_grouped_parameters1 = [
    {'params': [p for n, p in prompt_model.plm.named_parameters() if not any(nd in n for nd in no_decay)],
     'weight_decay': 0.01},
    {'params': [p for n, p in prompt_model.plm.named_parameters() if any(nd in n for nd in no_decay)],
     'weight_decay': 0.0}
]

# Using different optimizer for prompt parameters and model parameters

# optimizer_grouped_parameters2 = [
#     {'params': , "lr":1e-1},
# ]
optimizer1 = AdamW(optimizer_grouped_parameters1, lr=3e-5)
optimizer2 = AdamW(prompt_model.verbalizer.parameters(), lr=kptw_lr)
# print(optimizer_grouped_parameters2)

tot_step = len(train_dataloader)  # args.gradient_accumulation_steps * args.max_epochs
scheduler1 = get_linear_schedule_with_warmup(
    optimizer1,
    num_warmup_steps=0, num_training_steps=tot_step)

# scheduler2 = get_linear_schedule_with_warmup(
#     optimizer2,
#     num_warmup_steps=0, num_training_steps=tot_step)
scheduler2 = None

tot_loss = 0
log_loss = 0
best_val_acc = 0
for epoch in range(epochs):
    tot_loss = 0
    prompt_model.train()
    for step, inputs in enumerate(train_dataloader):
        if use_cuda:
            inputs = inputs.cuda()
        logits = prompt_model(inputs)
        labels = inputs['label']
        loss = loss_func(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(prompt_model.parameters(), 1.0)
        tot_loss += loss.item()
        optimizer1.step()
        scheduler1.step()
        optimizer1.zero_grad()
        # if optimizer2 is not None:
        #     optimizer2.step()
        #     optimizer2.zero_grad()
        # if scheduler2 is not None:
        #     scheduler2.step()

    val_acc = evaluate(prompt_model, validation_dataloader, desc="Valid")
    if val_acc >= best_val_acc:
        torch.save(prompt_model.state_dict(), "./best_val.ckpt")
        best_val_acc = val_acc
    print("Epoch {}, val_acc {}".format(epoch, val_acc), flush=True)

prompt_model.load_state_dict(torch.load("./best_val.ckpt"))
prompt_model = prompt_model.cuda()
test_acc = evaluate(prompt_model, test_dataloader, desc="Test")

content_write = "=" * 20 + "\n"
content_write += f"Acc: {test_acc}"
content_write += "\n\n"

print(content_write)
