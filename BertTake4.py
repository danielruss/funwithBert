from tensorflow import keras
import bert
import numpy as np
from bert.tokenization.bert_tokenization import FullTokenizer
import timeit
import os

model_dir = "/Users/druss/Downloads/uncased_L-12_H-768_A-12"

print("... loading parameters...")
t0 = timeit.default_timer()
bert_params = bert.params_from_pretrained_ckpt(model_dir)
l_bert = bert.BertModelLayer.from_params(bert_params, name="bert")
print("... parameters loaded in ", (timeit.default_timer() - t0), " sec.")

print("... Building model ...")
t0 = timeit.default_timer()
max_seq_len = 128
l_input_ids = keras.layers.Input(shape=(max_seq_len,), dtype='int32')
l_token_type_ids = keras.layers.Input(shape=(max_seq_len,), dtype='int32')

# using the default token_type/segment id 0
# output: [batch_size, max_seq_len, hidden_size]
output = l_bert(l_input_ids)
model = keras.Model(inputs=l_input_ids, outputs=output)
model.build(input_shape=(None, max_seq_len))
print("... model built in ", (timeit.default_timer() - t0), " sec.")
model.summary()

# load weights...
print("... loading model weights ...")
t0 = timeit.default_timer()
bert_ckpt_file = os.path.join(model_dir, "bert_model.ckpt")
bert.load_stock_weights(l_bert, bert_ckpt_file)
print("... model weights loaded in ", (timeit.default_timer() - t0), " sec.")


# lets try to embed a few sentences...
# pred_sentences = [
#   "That movie was absolutely awful",
#   "The acting was a bit lacking",
#   "The film was creative and surprising",
#   "Absolutely fantastic!"
# ]
pred_sentences = ["Hi there"]

bert_ckpt_dir = model_dir
tokenizer = FullTokenizer(vocab_file=os.path.join(bert_ckpt_dir, "vocab.txt"))
pred_tokens = map(tokenizer.tokenize, pred_sentences)
pred_tokens = map(lambda tok: ["[CLS]"] + tok + ["[SEP]"], pred_tokens)
pred_token_ids = list(map(tokenizer.convert_tokens_to_ids, pred_tokens))

pred_token_ids = map(lambda tids: tids +
                     [0]*(max_seq_len-len(tids)), pred_token_ids)
pred_token_ids = np.array(list(pred_token_ids))

print('pred_token_ids shape:', pred_token_ids.shape)

res = model.predict(pred_token_ids).argmax(axis=-1)
print('           res shape:', res.shape)

for text, sentiment in zip(pred_sentences, res):
    print("   text:", text)
    print(" tokens:", pred_tokens)
    print("    ids:", pred_token_ids)
    print("    res:", res)
