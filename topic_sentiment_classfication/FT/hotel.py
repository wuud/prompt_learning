#encoding=utf-8
import torch
from datasets import load_from_disk, load_dataset
from transformers import BertTokenizer
from transformers import BertModel
from transformers import AdamW
from openprompt.utils.reproduciblity import set_seed
import data_gen as my_util

epoch = 20
# 主题分类batchsize设为4，情感分类batchsize设为2
batch_size = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = 'E://models//chinese-roberta-wwm-ext'
model_name = 'bert'
max_length = 128
num_classes = 2
learn_rate = 2e-3
# shot = 16
# seed = 145


for seed in [2, 3, 143, 144, 145]:
    for shot in [1, 4, 8, 16]:
        set_seed(seed)
        ### hotel

        my_dataset = my_util.get_hotel_data()
        train_file, val_file = my_util.few_shot_sample(seed, shot, my_dataset, 'hotel')
        test_file = my_util.trans_test_dataset(my_dataset, 'hotel')
        Dataset = my_util.HotelDataset


        dataset = Dataset(train_file, val_file, test_file, 'train')
        print(dataset)
        print(len(dataset))


        token = BertTokenizer.from_pretrained(model_path)

        # token = BertTokenizer.from_pretrained(model_name) # huggingface 仓库

        def collate_fn(data):
            sents = [i[0] for i in data]
            labels = [i[1] for i in data]


            data = token.batch_encode_plus(batch_text_or_text_pairs=sents,
                                           truncation=True,
                                           padding='max_length',
                                           max_length=max_length,
                                           return_tensors='pt',
                                           return_length=True)
            # print('-' * 100)
            # print(data['input_ids'][0])
            # print(token.decode(data['input_ids'][0]))
            #input_ids:
            #attention_mask:
            input_ids = data['input_ids']
            attention_mask = data['attention_mask']
            token_type_ids = data['token_type_ids']
            labels = torch.LongTensor(labels)

            #print(data['length'], data['length'].max())

            return input_ids, attention_mask, token_type_ids, labels



        loader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_size=batch_size,
                                             collate_fn=collate_fn,
                                             shuffle=True,
                                             drop_last=True)




        pretrained = BertModel.from_pretrained(model_path).to(device)
        # pretrained = BertModel.from_pretrained(model_name).to(device)

        #
        # for param in pretrained.parameters():
        #     param.requires_grad_(False)

        #
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(768, num_classes)

            def forward(self, input_ids, attention_mask, token_type_ids):

                out = pretrained(input_ids=input_ids,
                           attention_mask=attention_mask,
                           token_type_ids=token_type_ids)

                out = self.fc(out.last_hidden_state[:, 0])
        #         print("1: ", out)
                out = out.softmax(dim=1)
        #         print("2: ", out)

                return out


        model = Model().to(device)



        #
        optimizer = AdamW(model.parameters(), lr=learn_rate)
        criterion = torch.nn.CrossEntropyLoss()

        model.train()

        print('--------------start training-----------------')
        best_val_acc = 0
        for poch in range(epoch):
            tot_loss = 0
            idx = 0
            correct = 0
            total = 0
            # print('loder, ',loader)
            for i, (input_ids, attention_mask, token_type_ids,
                labels) in enumerate(loader):
                input_ids, attention_mask, token_type_ids, labels = input_ids.to(device), attention_mask.to(device), token_type_ids.to(device), labels.to(device)
                out = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
                # print('lables shape: ',labels.shape)
            #     print(out.shape)
                loss = criterion(out, labels)
                loss.backward()
                tot_loss += loss.item()
                optimizer.step()
                optimizer.zero_grad()
                out = out.argmax(dim=1)
                idx = i
                correct += (out == labels).sum().item()
                total += len(labels)
            val_acc = correct / total
            # if val_acc >= best_val_acc:
            #     torch.save(model.state_dict(), "./best_val.ckpt")
            #     best_val_acc = val_acc
            print("Epoch {}, average loss: {}, batch loss: {}, valid acc: {}".format(poch, tot_loss / (idx + 1), loss.item(), val_acc), flush=True)
            # print('correct : total: ', correct, total)

        def test():
            # best_model = model.load_state_dict(torch.load("./best_val.ckpt"))
            # print(type(torch.load("./best_val.ckpt")))
            # best_model = best_model.to(device)
            model.eval()
            correct = 0
            total = 0

            #
            loader_test = torch.utils.data.DataLoader(dataset=Dataset(train_file, val_file, test_file, 'test'),
                                                      batch_size=batch_size,
                                                      collate_fn=collate_fn,
                                                      shuffle=True,
                                                      drop_last=True)

            print('test dataset length : ',len(loader_test))
            for i, (input_ids, attention_mask, token_type_ids,
                    labels) in enumerate(loader_test):
                input_ids, attention_mask, token_type_ids, labels = input_ids.to(device), attention_mask.to(device), token_type_ids.to(device), labels.to(device)
                # if i == 50:
                #     break

        #         print(i)

                with torch.no_grad():
                    out = model(input_ids=input_ids,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids)

                out = out.argmax(dim=1)
                correct += (out == labels).sum().item()
                total += len(labels)

            acc = correct / total
            print('test accuracy : ', acc)
            return acc

        acc = test()
        with open(f'./res/hotel/seed{seed}_shot{shot}.txt', 'w') as f:
            f.write('seed = {}, shot = {}, acc = {}'.format(seed, shot, acc))