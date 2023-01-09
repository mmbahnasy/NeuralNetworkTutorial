import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# lstm = nn.LSTM(3, 3)  # Input dim is 3, output dim is 3
# inputs = [torch.randn(1, 3) for _ in range(5)]
# # print(inputs)
# hidden = (torch.rand(1, 1, 3), torch.rand(1, 1, 3))
# # for i in inputs:
# #     out, hidden = lstm(i.view(1, 1, 3), hidden)

# out, hidden = lstm(torch.cat(inputs).view(-1, 1, 3), hidden)

# print("hidded:", hidden)
# print("Out:", out)

training_data = [
    "A lovely day, isn’t it".split(),
    "Do I have to".split(),
    "Can I help you".split(),
    "How are things going".split(),
    "Any thing else".split(),
    "Are you kidding".split(),
    "Are you sure".split(),
    "Do you understand me".split(),
    "Are you done".split(),
    "Can I ask you something".split(),
    "Can you please repeat that".split(),
    "Did you get it".split(),
]

# print(training_data)

word_to_ix = {}
for sent in training_data:
    for word in sent:
        if word not in word_to_ix:  # word has not been assigned an index yet
            word_to_ix[word] = len(word_to_ix)  # Assign each word with a unique index
print("len(word_to_ix):", len(word_to_ix))

HOTVECTOR = F.one_hot(torch.arange(len(word_to_ix)))

# print(HOTVECTOR)
def prepare_sequence(seq, to_ix=word_to_ix):
    idxs = [HOTVECTOR[to_ix[w]].reshape(1,-1) for w in seq]
    return torch.cat(idxs, dim=0).float()

# print(prepare_sequence(training_data[2]))

EMBEDDING_DIM = HIDDEN_DIM = len(word_to_ix)
class LSTMTagger(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.rnn = nn.GRU(embedding_dim, hidden_dim)

    def forward(self, sentence):
        lstm_out, _ = self.rnn(sentence.view(-1, 1, self.embedding_dim))
        return lstm_out

model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)
MODEL_PATH = "./lstm_wordprediction.pt"
try:
    model.load_state_dict(torch.load(MODEL_PATH))
    print("Model parameters were loaded from:", MODEL_PATH)
except:
    pass

with torch.no_grad():
    inputs = prepare_sequence(training_data[0], word_to_ix)
    print("inputs.shape:", inputs.shape)
    output = model(inputs[:-1,:]) # Feed the model of N-1 words and expect the last word
    print("Expected word index:", inputs[-1,:].argmax(0))
    print("Output word index:", output[-1,:].argmax(1))

for epoch in range(300):
    for sentence in training_data:
        model.zero_grad()
        sentence_in = prepare_sequence(sentence, word_to_ix)

        sentence_out = model(sentence_in[:-1,:]) # input 0 to N-1 words, and train the model to generate 1 to N
        labels = sentence_in[1:,:].argmax(1)
        loss = loss_function(sentence_out.view(len(sentence_out),-1), labels) # Train the model to generate 1 to N
        loss.backward()
        optimizer.step()
    print(f"loss: {loss:>7f}")

torch.save(model.state_dict(), MODEL_PATH)

with torch.no_grad():
    correctSeq = 0
    for sentence in training_data:
        inputs = prepare_sequence(sentence, word_to_ix)
        output = model(inputs[:-1,:]) # Feed the model of N-1 words and expect the last word
        # print("Expected word index:", inputs[-1,:].argmax(0))
        # print("Output word index:", output[-1,:].argmax(1))
        correctSeq +=1 if inputs[-1,:].argmax(0) == output[-1,:].argmax(1) else 0
    print("Accuracy:", correctSeq / len(training_data))

