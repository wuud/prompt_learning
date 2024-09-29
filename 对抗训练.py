import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW

# 设置设备，如果有可用的GPU则使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载预训练的BERT模型和tokenizer
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 将模型移动到设备上
model.to(device)

# 定义生成对抗样本的函数，这里以添加扰动为例
def generate_adversarial_sample(text):
    encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    input_ids = encoded_input['input_ids'].to(device)
    attention_mask = encoded_input['attention_mask'].to(device)
    labels = torch.tensor([1]).unsqueeze(0).to(device)  # 此处示例中仅有一个样本，并标记为类别1

    # 生成对抗样本
    perturbed_input_ids = input_ids + 0.1 * torch.randn_like(input_ids).to(device)
    perturbed_input_ids = perturbed_input_ids.clamp(0, tokenizer.vocab_size - 1)

    # 计算对抗样本的损失
    outputs = model(perturbed_input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss

    return perturbed_input_ids, loss

# 定义判别器模型
discriminator = torch.nn.Sequential(
    torch.nn.Linear(model.config.hidden_size, 1),
    torch.nn.Sigmoid()
)

# 将判别器移动到设备上
discriminator.to(device)

# 定义优化器
optimizer = AdamW(model.parameters(), lr=1e-5)
discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-3)

# 对抗训练的轮数和步数
num_epochs = 5
steps_per_epoch = 100

# 对抗训练循环
for epoch in range(num_epochs):
    for step in range(steps_per_epoch):
        # 生成对抗样本
        text = "This is a sample sentence for classification."
        perturbed_input_ids, loss = generate_adversarial_sample(text)

        # 更新生成器
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 更新判别器
        discriminator_optimizer.zero_grad()
        real_outputs = discriminator(model(input_ids, attention_mask=attention_mask).pooler_output)
        perturbed_outputs = discriminator(model(perturbed_input_ids, attention_mask=attention_mask).pooler_output)
        discriminator_loss = torch.nn.BCELoss()(
            torch.cat((real_outputs, perturbed_outputs)),
            torch.cat((torch.ones_like(real_outputs), torch.zeros_like(perturbed_outputs)))
        )
        discriminator_loss.backward()
        discriminator_optimizer.step()