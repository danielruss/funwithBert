import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import preprocessing
from transformers import (TFBertForSequenceClassification,
                          BertTokenizer)
from tqdm import tqdm

print('numpy version: ', np.__version__)
print('pandas version: ', pd.__version__)
print('tenserflow version: ', tf.__version__)

y = ['dog', 'dog', 'cat', 'dog', 'cat', 'cat']
print(y)
labelEncoder = preprocessing.LabelEncoder()
labelEncoder.fit(y)
print(labelEncoder.classes_)
y_transform = labelEncoder.transform(y)
print(y_transform)
print(labelEncoder.inverse_transform(y_transform))
