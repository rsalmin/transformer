# -*- coding: utf-8 -*-
import tensorflow_datasets as tfds
import inputData as data

tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(
                     (en.numpy() for en in data.en_train), target_vocab_size=2**15)
tokenizer_en.save_to_file('data/tokenizer_en')

tokenizer_ru = tfds.features.text.SubwordTextEncoder.build_from_corpus(
                     (ru.numpy() for ru in data.ru_train), target_vocab_size=2**15)
tokenizer_ru.save_to_file('data/tokenizer_ru')
