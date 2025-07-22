import numpy as np
import random
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from nltk_utils import bag_of_words, tokenize, stem
from model import NeuralNet

with open('intents.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []
# lặp qua từng câu trong các mẫu ý định của chúng tôi
for intent in intents['intents']:
    tag = intent['tag']
    # thêm vào danh sách thẻ
    tags.append(tag)
    for pattern in intent['patterns']:
        # phân tách từng từ trong câu
        w = tokenize(pattern)
        # thêm vào danh sách từ của chúng tôi
        all_words.extend(w)
        # thêm vào cặp xy
        xy.append((w, tag))

# gốc và chuyển đổi thành chữ thường mỗi từ
ignore_words = ['?', '.', '!']
all_words = [stem(w) for w in all_words if w not in ignore_words]
# loại bỏ các từ trùng lặp và sắp xếp
all_words = sorted(set(all_words))
tags = sorted(set(tags))

print(len(xy), "mẫu")
print(len(tags), "thẻ:", tags)
print(len(all_words), "từ gốc duy nhất:", all_words)

# tạo dữ liệu huấn luyện
X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    # X: túi từ cho mỗi pattern_sentence
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    # y: PyTorch CrossEntropyLoss chỉ cần nhãn lớp, không phải one-hot
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Siêu tham số 
num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)
print(input_size, output_size)

class ChatDataset(Dataset):

    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    # hỗ trợ lập chỉ mục sao cho dataset[i] có thể được sử dụng để lấy mẫu thứ i
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # chúng ta có thể gọi len(dataset) để trả về kích thước
    def __len__(self):
        return self.n_samples

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Mất mát và tối ưu hóa
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Huấn luyện mô hình
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        
        # Truyền qua
        outputs = model(words)
        # nếu y sẽ là one-hot, chúng ta phải áp dụng
        # labels = torch.max(labels, 1)[1]
        loss = criterion(outputs, labels)
        
        # Ngược và tối ưu hóa
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch+1) % 100 == 0:
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


print(f'mất mát cuối cùng: {loss.item():.4f}')

data = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"all_words": all_words,
"tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'Huấn luyện hoàn tất. tệp đã được lưu vào {FILE}')
