from tensorflow import keras
import os
from tensorflow.keras import layers
import tensorflow_hub as hub
import tensorflow as tf

import pandas as pd
import numpy as np
import bert

import re
import random
import math
import hashlib
import time
import matplotlib
import copy

t0 = time.process_time()

TAG_RE = re.compile(r'<[^>]+>')


def remove_tags(text):
    return TAG_RE.sub('', text)


def preprocess_text(sen):
    # Removing html tags
    sentence = remove_tags(sen)

    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)

    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)

    return sentence


t1 = time.process_time()
print("time: ", (t1-t0), " sec")

t0 = time.process_time()

# load the data
movie_reviews = pd.read_csv("/Users/druss/Downloads/IMDB Dataset.csv")
movie_reviews.shape

# preprocess the data
reviews = []
sentences = list(movie_reviews['review'])
for sen in sentences:
    reviews.append(preprocess_text(sen))

# get the labels...
y = np.array(
    list(map(lambda x: 1 if x == "positive" else 0, movie_reviews['sentiment'])))

# tokenzize the data...
BertTokenizer = bert.bert_tokenization.FullTokenizer
bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",
                            trainable=False)
vocabulary_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
to_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = BertTokenizer(vocabulary_file, to_lower_case)

# create a function for tokenizing reviews...


def tokenize_reviews(text_reviews):
    return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text_reviews))


tokenized_reviews = [tokenize_reviews(review) for review in reviews]

random.seed(4)
reviews_with_len = [[review, y[i], len(review)]
                    for i, review in enumerate(tokenized_reviews)]
random.shuffle(reviews_with_len)

reviews_with_len.sort(key=lambda x: x[2])
sorted_reviews_labels = [(review_lab[0], review_lab[1])
                         for review_lab in reviews_with_len]

#processed_dataset = tf.data.Dataset.from_generator(lambda: sorted_reviews_labels, output_types=(tf.int32, tf.int32))
#BATCH_SIZE = 32
#batched_dataset = processed_dataset.padded_batch(BATCH_SIZE, padded_shapes=((None, ), ()))

#TOTAL_BATCHES = math.ceil(len(sorted_reviews_labels) / BATCH_SIZE)
#TEST_BATCHES = TOTAL_BATCHES // 10
# batched_dataset.shuffle(TOTAL_BATCHES)
#test_data = batched_dataset.take(TEST_BATCHES)
#train_data = batched_dataset.skip(TEST_BATCHES)


zz = pd.DataFrame({
    "text": pd.Series(reviews),
    "tokens": pd.Series(copy.deepcopy(tokenized_reviews)),
    "label": pd.Series(y)
})
zz["len"] = [len(x) for x in zz["tokens"]]

# split the data into Train/test based on the md5 hash...
zz["hash"] = [hashlib.md5(x.encode('ascii')).hexdigest() for x in zz["text"]]
zz["Test"] = [(int(x, 16) % 100 > 90) for x in zz["hash"]]
train_data = copy.deepcopy(zz[zz["Test"] != True])
test_data = copy.deepcopy(zz[zz["Test"] == True])
print(train_data.shape)
print(test_data.shape)

t1 = time.process_time()
print("time: ", (t1-t0), " sec")


def addSpecialTokens(lst, maxValues):
    lst = lst[:maxValues-2]
    lst.insert(0, 101)
    lst.append(102)
    return lst


#test_data["tokens"].apply(lambda x:x.insert(0,1))
test_data["tokens"] = test_data["tokens"].apply(addSpecialTokens, args=(300,))
train_data["tokens"] = train_data["tokens"].apply(
    addSpecialTokens, args=(300,))

# print the values out ....
print("tokenized_reviews:",
      tokenized_reviews[6][0:10], "len:", len(tokenized_reviews[6]))
print("zz:", zz.iloc[6, 1][0:10], "len:", len(zz.iloc[6, 1]))
print("test_data:", test_data.iloc[0, 1]
      [0:10], "len:", len(test_data.iloc[0, 1]))

print("train_data:", train_data.iloc[0, 1]
      [0:10], "len:", len(train_data.iloc[0, 1]))

t0 = time.process_time()
model_dir = "/Users/druss/Downloads/uncased_L-12_H-768_A-12"

bert_params = bert.params_from_pretrained_ckpt(model_dir)
l_bert = bert.BertModelLayer.from_params(bert_params, name="bert")

t1 = time.process_time()
print("time: ", (t1-t0), " sec")


t0 = time.process_time()


max_seq_len = 300
l_input_ids = keras.layers.Input(shape=(max_seq_len,), dtype='int32')
l_token_type_ids = keras.layers.Input(shape=(max_seq_len,), dtype='int32')

# using the default token_type/segment id 0
# output: [batch_size, max_seq_len, hidden_size]
output = l_bert(l_input_ids)
model = keras.Model(inputs=l_input_ids, outputs=output)
model.build(input_shape=(None, max_seq_len))

bert_ckpt_file = os.path.join(model_dir, "bert_model.ckpt")
bert.load_stock_weights(l_bert, bert_ckpt_file)

t1 = time.process_time()
print("time: ", (t1-t0), " sec")


def get_masks(tokens, max_seq_length):
    return [1]*len(tokens) + [0] * (max_seq_length - len(tokens))


def get_segments(tokens, max_seq_length):
    return [0] * (max_seq_length)


def pad_tokens(tokens, max_seq_length):
    padded = tokens + [0]*(max_seq_length - len(tokens))
    return padded


MAX_SEQ_LEN = 300
# before we pad the data, make a mask and segments, which are all 0 in our case ...
test_data["mask"] = test_data.iloc[:, 1].apply(get_masks, args=(MAX_SEQ_LEN,))
train_data["mask"] = train_data.iloc[:, 1].apply(
    get_masks, args=(MAX_SEQ_LEN,))

# one of bert's training has does sentence 2 follow sentence 1.
# we are not useing that method, so just fill the array with 0.
test_data["segments"] = test_data.iloc[:, 1].apply(
    get_segments, args=(MAX_SEQ_LEN,))
train_data["segments"] = train_data.iloc[:, 1].apply(
    get_segments, args=(MAX_SEQ_LEN,))


# Pad the data (Tokens only)...
test_data.iloc[:, 1] = test_data.iloc[:, 1].apply(
    pad_tokens, args=(MAX_SEQ_LEN,))
train_data.iloc[:, 1] = train_data.iloc[:, 1].apply(
    pad_tokens, args=(MAX_SEQ_LEN,))


input_tokens = test_data.loc[:, "tokens"].tolist()
input_masks = test_data.loc[:, "mask"].tolist()
input_segments = test_data.loc[:, "segments"].tolist()

pool_embs, all_embs = model.predict(
    [[input_tokens][0], [input_masks][0], [input_segments][0]])
