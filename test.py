import math
import torchtext
import torch
import torch.nn as nn
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import vocab
from torchtext.utils import download_from_url, extract_archive
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import time
import numpy as np
from model import Transformer
from nltk.translate.bleu_score import sentence_bleu


url_base = 'https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/'
train_urls = ('train.de.gz', 'train.en.gz')
test_urls = ('test_2016_flickr.de.gz', 'test_2016_flickr.en.gz')

train_filepaths = [extract_archive(download_from_url(url_base + url))[0] for url in train_urls]
test_filepaths = [extract_archive(download_from_url(url_base + url))[0] for url in test_urls]

de_tokenizer = get_tokenizer('spacy', language='de_core_news_sm')
en_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')

def build_vocab(path, tokenizer):
    counter = Counter()
    with open(path, encoding="utf8") as f:
        for sentence in f:
            counter.update(tokenizer(sentence))
    built_vocab = vocab(counter, specials=['<unk>', '<pad>', '<bos>', '<eos>'])
    built_vocab.set_default_index(built_vocab['<unk>'])
    return built_vocab

de_vocab = build_vocab(train_filepaths[0], de_tokenizer)
en_vocab = build_vocab(train_filepaths[1], en_tokenizer)

def data_process(paths):
    de_data = open(paths[0], encoding="utf8").readlines()
    en_data = open(paths[1], encoding="utf8").readlines()
    data = []
    for de_sentence, en_sentence in zip(de_data, en_data):
        de_tensor = torch.tensor([de_vocab[token] for token in de_tokenizer(de_sentence.rstrip("\n"))], dtype=torch.long)
        en_tensor = torch.tensor([en_vocab[token] for token in en_tokenizer(en_sentence.rstrip("\n"))], dtype=torch.long)
        data.append((de_tensor, en_tensor))
    return data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 128
PAD_IDX = de_vocab['<pad>']
BOS_IDX = de_vocab['<bos>']
EOS_IDX = de_vocab['<eos>']

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask
def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

MODEL_PATH = "model.pt"
transformer = torch.load(MODEL_PATH)
transformer = transformer.to(device)

def get_bleu_dataset(paths):
    de_data = open(paths[0], encoding="utf8").readlines()
    en_data = open(paths[1], encoding="utf8").readlines()
    data = []
    for de_sentence, en_sentence in zip(de_data, en_data):
        de_processed_sentence = de_sentence.rstrip("\n")
        en_processed_sentence = en_sentence.rstrip("\n")
        data.append((de_processed_sentence, en_processed_sentence))
    return data

def greedy_search(model, src, src_mask, max_len, start_symbol):
    src = src.to(device)
    src_mask = src_mask.to(device)
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
    for i in range(max_len-1):
        memory = memory.to(device)
        memory_mask = torch.zeros(ys.shape[0], memory.shape[0]).to(device).type(torch.bool)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0)).type(torch.bool)).to(device)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.fc_out(out[:, -1])
        _, next_word = torch.max(prob, dim = 1)
        next_word = next_word.item()

        ys = torch.cat([ys,torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == EOS_IDX:
            break
    return ys

def translate(model, src, src_vocab, tgt_vocab, src_tokenizer):
    model.eval()
    tokens = [BOS_IDX] + [src_vocab[tok] for tok in src_tokenizer(src)] + [EOS_IDX]
    num_tokens = len(tokens)
    src = (torch.LongTensor(tokens).reshape(num_tokens, 1))
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_search(model, src, src_mask, max_len=num_tokens + 5, start_symbol=BOS_IDX).flatten()
    return " ".join([tgt_vocab.get_itos()[tok] for tok in tgt_tokens]).replace("<bos>", "").replace("<eos>", "")

bleu_scores = []
bleu_testdata = get_bleu_dataset(test_filepaths)
for src, trg in bleu_testdata:
    pred = translate(transformer, src, de_vocab, en_vocab, de_tokenizer)
    bleu_scores.append(sentence_bleu([trg], pred, weights=(1, 0, 0, 0)))
print(sum(bleu_scores)/len(bleu_scores))