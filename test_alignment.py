import pandas as pd
from SpeechRepresentation.Utils import align_reward_func, align_seq

r1 = align_reward_func(
    'word',
    'word2'
)
assert r1 == -1

r2 = align_reward_func(
    'word',
    'wor'
)
assert r2 == 0

r3 = align_reward_func(
    'word',
    'word'
)
assert r3 == 1

dataframe = pd.read_csv('phonemes1.txt')
words = dataframe['word'].tolist()
