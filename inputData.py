# -*- coding: utf-8 -*-
import tensorflow as tf
import os

dataPath = 'data'

en_train = tf.data.TextLineDataset(os.path.join(dataPath, 'en.train'))
ru_train = tf.data.TextLineDataset(os.path.join(dataPath, 'ru.train'))
train_examples = tf.data.Dataset.zip((ru_train, en_train))

en_valid = tf.data.TextLineDataset(os.path.join(dataPath, 'en.dev'))
ru_valid = tf.data.TextLineDataset(os.path.join(dataPath, 'ru.dev'))
valid_examples = tf.data.Dataset.zip((ru_valid, en_valid))
