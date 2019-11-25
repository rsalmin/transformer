import tensorflow_datasets as tfds
import tensorflow as tf

import inputData

import numpy as np
import os.path

dataPath = 'data'
trainDatasetPath = os.path.join(dataPath, 'train_dataset')
valDatasetPath = os.path.join(dataPath, 'val_dataset')

BUFFER_SIZE = 20000
BATCH_SIZE = 64
"""Note: To keep this example small and relatively fast, drop examples with a length of over 40 tokens."""

MAX_LENGTH = 40


tokenizer_en = tfds.features.text.SubwordTextEncoder.load_from_file('data/tokenizer_en')
tokenizer_ru  = tfds.features.text.SubwordTextEncoder.load_from_file('data/tokenizer_ru')

sample_string = 'Transformer is awesome.'

tokenized_string = tokenizer_en.encode(sample_string)
print ('Tokenized string is {}'.format(tokenized_string))

original_string = tokenizer_en.decode(tokenized_string)
print ('The original string: {}'.format(original_string))

assert original_string == sample_string

"""The tokenizer encodes the string by breaking it into subwords if the word is not in its dictionary."""

for ts in tokenized_string:
  print ('{} ----> {}'.format(ts, tokenizer_en.decode([ts])))

"""Add a start and end token to the input and target."""

def encode(lang1, lang2):
  lang1 = [tokenizer_ru.vocab_size] + tokenizer_ru.encode(
      lang1.numpy()) + [tokenizer_ru.vocab_size+1]

  lang2 = [tokenizer_en.vocab_size] + tokenizer_en.encode(
      lang2.numpy()) + [tokenizer_en.vocab_size+1]

  return lang1, lang2

input_vocab_size = tokenizer_ru.vocab_size + 2
target_vocab_size = tokenizer_en.vocab_size + 2

def filter_max_length(x, y, max_length=MAX_LENGTH):
  return tf.logical_and(tf.size(x) <= max_length,
                        tf.size(y) <= max_length)

"""Operations inside `.map()` run in graph mode and receive a graph tensor that do not have a numpy attribute. The `tokenizer` expects a string or Unicode symbol to encode it into integers. Hence, you need to run the encoding inside a `tf.py_function`, which receives an eager tensor having a numpy attribute that contains the string value."""

def tf_encode(ru, en):
  return tf.py_function(encode, [ru, en], [tf.int64, tf.int64])

train_dataset = inputData.train_examples.map(tf_encode)
train_dataset = train_dataset.filter(filter_max_length)
# cache the dataset to memory to get a speedup while reading from it.
train_dataset = train_dataset.cache()
train_dataset = train_dataset.shuffle(BUFFER_SIZE).padded_batch(
    BATCH_SIZE, padded_shapes=([-1], [-1]))
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)


val_dataset = inputData.valid_examples.map(tf_encode)
val_dataset = val_dataset.filter(filter_max_length).padded_batch(
    BATCH_SIZE, padded_shapes=([-1], [-1]))

#writer_train = tf.data.experimental.TFRecordWriter(trainDatasetPath)
#writer_train.write(train_dataset_ru)

#writer_val = tf.data.experimental.TFRecordWriter(valDatasetPath)
#writer_val.write(val_dataset)
