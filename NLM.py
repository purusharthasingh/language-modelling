# -*- coding:utf8 -*-

"""
This py page is for the Modeling and training part of this NLM. 
Try to edit the place labeled "# TODO"!!!
"""

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import random


nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

CONTEXT_SIZE = 2
EMBEDDING_DIM = 50

trigrams = []
vocab = {}


def word2index(word, vocab):
    """
    Convert an word token to an dictionary index
    """
    if word in vocab:
        value = vocab[word]
    else:
        value = -1
    return value


def index2word(index, vocab):
    """
    Convert an word index to a word token
    """
    for w, v in vocab.items():
        if v == index:
            return w
    return 0


def preprocess(file, is_filter=True):
    """
    Prepare the data and the vocab for the models.
    For expediency, the vocabulary will be all the words
    in the dataset (not split into training/test), so
    the assignment can avoid the OOV problem.
    """
    with open(file, 'r') as fr:
        for idx, line in enumerate(fr):
            words = word_tokenize(line)
            if is_filter:
                words = [w for w in words if not w in stop_words]
                words = [word.lower() for word in words if word.isalpha()]
                for word in words:
                    if word not in vocab:
                        vocab[word] = len(vocab)
            if len(words) > 0:
                for i in range(len(words) - 2):
                    trigrams.append(([words[i], words[i + 1]], words[i + 2]))
    print('{0} contain {1} lines'.format(file, idx + 1))
    print('The size of dictionary is：{}'.format(len(vocab)))
    print('The size of trigrams is：{}'.format(len(trigrams)))
    return 0


class NgramLM(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NgramLM, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 256)
        self.linear2 = nn.Linear(256, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        # TODO
        out = self.linear1(embeds)
        out = F.relu(out)
        # out = F.dropout(out, 0.1)
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs


def output(model, file1, file2):
    """
    Write the embedding file to the disk
    """
    print("here")
    with open(file1, 'w', encoding='utf-8') as fw1:
        for word, id in vocab.items():
            # TODO
            # embed = model.embeddings.weight(id)
            # embed_2 = model.embeddings(id)
            # embed_3 = model.embeddings(torch.tensor([id])).view((1, -1))
            emb = (model.embeddings(torch.tensor([id]))[0]).tolist()
            ostr = str(word)
            emb_s = ''
            for x in emb:
                emb_s += (' '+str(x))
            # emb = ' '.join(emb)
            ostr += emb_s + '\n'
            fw1.write(ostr)
    with open(file2, 'w', encoding='utf-8') as fw2:
        for word,id in vocab.items():
            # TODO
            emb = torch.rand(50).tolist()
            ostr = str(word)
            emb_s = ''
            for x in emb:
                emb_s += (' ' + str(x))
            # emb = ' '.join(emb)
            ostr += emb_s + '\n'
            fw2.write(ostr)


def training():
    """
    Train the NLM
    """
    preprocess('./data/reviews_500.txt')
    losses = []
    inpts = [x[0] for x in trigrams]
    for i, context in enumerate(inpts):
        ids = [word2index(word, vocab) for word in context]
        inpts[i] = ids

    trgts = [word2index(x[1], vocab) for x in trigrams]
    dataset = torch.utils.data.TensorDataset(torch.tensor(inpts), torch.tensor(trgts))
    # batches = torch.utils.data.DataLoader(dataset, batch_size=1,)

    model = NgramLM(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)

    # TODO
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.8)
    loss_function = torch.nn.NLLLoss()
    for epoch in range(5):
        total_loss = 0
        print(epoch)
        # batches_sample = random.sample(list(batches), len(trigrams)//1.5)
        # subset_trigrams = random.sample(trigrams, len(trigrams)//4)
        i=0
        batches = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True,)
        for context, target in batches:
            i+=1
            if i % 9 == 0:
                continue
            # Step 1. Prepare the inputs to be passed to the model (i.e, turn the words
            # into integer indices and wrap them in tensors)
            # ids = [word2index(word, vocab) for word in context]
            # ids_tensors = torch.tensor(ids, dtype=torch.long)
            # TODO
            # Step 2. Recall that torch *accumulates* gradients. Before passing in a
            # new instance, you need to zero out the gradients from the old instance
            # TODO
            model.zero_grad()
            # Step 3. Run the forward pass, getting log probabilities over next
            # words
            # TODO

            log_p = model(context)
            # Step 4. Compute your loss function. (Again, Torch wants the target
            # word wrapped in a tensor)
            # target_id = torch.tensor([word2index(target, vocab)] ,dtype=torch.long)
            loss = loss_function(log_p, target)
            # TODO
            # assert (0 == 1)
            # Step 5. Do the backward pass and update the gradient
            loss.backward()
            optimizer.step()
            # TODO
            # Get the Python number from a 1-element Tensor by calling tensor.item()
            total_loss += loss.item()
        losses.append(total_loss)
    print(losses)  # The loss decreased every iteration over the training data!
    # output('./data/embedding.txt_100', './data/embedding_random.txt_100')
    output(model, './data/embedding.txt', './data/random_embedding.txt')


if __name__ == '__main__':
    t1 = time.time()
    training()
    t2 = time.time()
    print("Total time: " + str((t2-t1)/60))
