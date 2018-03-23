#!/usr/bin/python
# -*- coding: utf-8 -*-
# encoding: UTF-8

from collections import Counter
import tensorflow.contrib.keras as kr
import numpy as np
import os
import jieba
import jieba.analyse
from gensim.models.keyedvectors import KeyedVectors


def open_file(filename, mode='r'):
    """
    Commonly used file reader, change this to switch between python2 and python3.
    mode: 'r' or 'w' for read or write
    """
    return open(filename, mode, encoding='utf-8', errors='ignore')

def read_file(filename):
    """读取文件数据"""
    contents, labels = [], []
    with open_file(filename) as f:
        for line in f:
            try:
                label, content = line.strip().split('\t')
                # contents.append(list(content))
                labels.append(label)
                # 把这里替换成分词的
                temp=jieba.analyse.extract_tags(line, topK=156, withWeight=False, allowPOS=())
                temp.extend(['padding'] * (156 - len(temp))) if len(temp) < 156 else None
                contents.append(temp)

            except:
                pass
    return contents, labels

def build_vocab(train_dir, vocab_dir, times_dir,vocab_size=5000):
    """根据训练集构建词汇表，存储"""
    data_train, _ = read_file(train_dir)

    all_data = []
    for content in data_train:
        all_data.extend(content)

    counter = Counter(all_data)
    count_pairs = counter.most_common()  # 返回一个TopN列表c.most_common(3)  [('a', 5), ('r', 2), ('b', 2)]
    s0 = count_pairs[4998]
    s = count_pairs[4999]
    s1 = count_pairs[5000]
    s2 = count_pairs[5001]
    s2 = count_pairs[5002]
    s3 = count_pairs[5003]
    words, _ = list(zip(*count_pairs))
    # 添加一个 <PAD> 来将所有文本pad为同一长度
    words = ['<PAD>'] + list(words)
    _ = ['<PAD>'] + list(str(x) for x in _)

    # print('找数据规律',count_pairs[4999],count_pairs[5000],count_pairs[5001],count_pairs[5002],count_pairs[5003],count_pairs[5004])

    open_file(vocab_dir, mode='w').write('\n'.join(words) + '\n')
    # open_file(times_dir, mode='w').write('\n'.join(_) + '\n')
def read_vocab(vocab_dir):
    """读取词汇表"""
    words = open_file(vocab_dir).read().strip().split('\n')
    word_to_id = dict(zip(words, range(len(words))))

    return words, word_to_id

def read_category():
    """读取分类目录，固定"""
    categories = ['体育', '财经', '彩票', '社会', '股票', '房产', '家居',
                  '教育', '科技', '时尚', '时政', '游戏', '娱乐']
    # categories = ['体育', '财经', '房产', '家居',
    #     '教育', '科技', '时尚', '时政', '游戏', '娱乐']
    cat_to_id = dict(zip(categories, range(len(categories))))

    return categories, cat_to_id


def to_words(content, words):
    """将id表示的内容转换为文字"""
    return ''.join(words[x] for x in content)


# process_file(test_dir, word_to_id, cat_to_id, 'padding', word2vec_dir)
def process_file(filename, cat_to_id, padding_token, file_to_load=None, max_length=None):
    """将文件转换为id表示"""
    contents, labels = read_file(filename)

    data_id, label_id = [], []
    # max_sentence_length = max_length if max_length is not None else max([len(sentence) for sentence in contents])
    # for sentence in contents:
    #     if len(sentence) > max_sentence_length:
    #         # sentence = sentence[:max_sentence_length]
    #         del sentence[max_sentence_length:]
    #
    #     else:
    #         sentence.extend([padding_token] * (max_sentence_length - len(sentence)))
    for i in range(len(contents)):
        # data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
        # 处理成word2vec形式
        label_id.append(cat_to_id[labels[i]])

    data_id = embedding_sentences(contents, file_to_load)

    # 使用keras提供的pad_sequences来将文本pad为固定长度
    # x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length)
    y_pad = kr.utils.to_categorical(label_id) # 将标签转换为one-hot表示

    return data_id, y_pad


def embedding_sentences(sentences, file_to_load = None):
    if file_to_load is not None:
         w2vModel = KeyedVectors.load_word2vec_format(file_to_load, binary=True)
        # w2vModel = Word2Vec.load(file_to_load)

    all_vectors = []
    embeddingDim = w2vModel.vector_size
    embeddingUnknown = [0 for i in range(embeddingDim)]
    for sentence in sentences:
        this_vector = []
        for word in sentence:
            if word in w2vModel.vocab:
                 this_vector.append(w2vModel[word])
            else:
                this_vector.append(embeddingUnknown)
        all_vectors.append(this_vector)
    return all_vectors

def batch_iter(x, y, batch_size=64):
    """生成批次数据"""
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1
    x = np.array(x)
    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]
