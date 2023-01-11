# Source: https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

# word_to_ix = {"hello": 0, "world": 1}
# embeds = nn.Embedding(2, 5)  # 2 words in vocab, 5 dimensional embeddings
# lookup_tensor = torch.tensor([word_to_ix["world"]], dtype=torch.long)
# print("lookup_tensor:", lookup_tensor)
# embed = embeds(lookup_tensor)
# print("Embedding:", embed)

CONTEXT_SIZE = 2
EMBEDDING_DIM = 10
# We will use Shakespeare Sonnet 2
test_sentence = """In the Name of Allah the Most Compassionate, Most Merciful.
All praise is for Allah-Lord of all worlds,
the Most Compassionate, Most Merciful,
Master of the Day of Judgment,
You alone we worship and You alone we ask for help,
Guide us along the Straight Path,
the Path of those You have blessed—not those You are displeased with, or those who are astray.""".split()
# build a list of tuples.
# Each tuple is ([ word_i-CONTEXT_SIZE, ..., word_i-1 ], target word)
ngrams = [
    (
        [test_sentence[i - j - 1] for j in range(CONTEXT_SIZE)],
        test_sentence[i]
    )
    for i in range(CONTEXT_SIZE, len(test_sentence))
]
# Print the first 3, just so you can see what they look like.
print(ngrams[:3])

vocab = set(test_sentence)
word_to_ix = {word: i for i, word in enumerate(vocab)}

class NGramLanguageModeler(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs

loss_function = nn.NLLLoss()
model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
optimizer = optim.SGD(model.parameters(), lr=0.001)
losses = []

for epoch in range(1000):
    total_loss = 0
    for context, target in ngrams:
        context_idxs = torch.tensor([word_to_ix[w] for w in context], dtype=torch.long)
        model.zero_grad()
        log_probs = model(context_idxs)
        # print("Loss to be applied on:\n", log_probs.shape, "\n and \n", torch.tensor([word_to_ix[target]], dtype=torch.long))
        loss = loss_function(log_probs, torch.tensor([word_to_ix[target]], dtype=torch.long))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    losses.append(total_loss)
    if epoch%100==99: print(total_loss)

correct_prediction = 0
for context, target in ngrams:
    context_idxs = torch.tensor([word_to_ix[w] for w in context], dtype=torch.long)
    with torch.no_grad():
        log_probs = model(context_idxs)
        # print("log_probs", log_probs.shape, log_probs.argmax(dim=1), word_to_ix[target])
        correct_prediction+= 1 if log_probs.argmax(dim=1) == word_to_ix[target] else 0
print("Correct prediction:", correct_prediction/len(ngrams) * 100, "%")
