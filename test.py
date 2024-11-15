# -*- coding: utf-8 -*-
import os
from SpeechRepresentation.Semantic import SemanticDissimilarity, LexicalSurprisal
from SpeechRepresentation.Utils import CWordStim
import numpy as np
def test_dissimilarity_impulses(modelName = 'glove-wiki-gigaword-100'):
    sentences = None
    with open('text1.txt', 'r') as f:
        sentences = f.readlines() #here I reorganized text into lines of single sentence
        for idx in range(len(sentences)):
            text = sentences[idx]
            if os.name == 'nt':
                text = text.replace('\r\n',' ').replace('\n',' ').replace('-',' ')
            else:
                text = text.replace('\n',' ').replace('-',' ')
            sentences[idx] = text
    
    
    oModel = SemanticDissimilarity.CDissimilarityVector(modelName)
    words, vectors = oModel.get(sentences)
    oStim = CWordStim()
    oStim.loadWordTiming('phonemes1.txt')
    oStim.lowerWords()
    oStim.alignVecToWordTiming(words, vectors)
    impulses = oStim.toImpulses(f = 64)
    return impulses,vectors
    

def test_lexical_surprisal():
    sentences = None
    with open('text1.txt', 'r') as f:
        sentences = f.readlines() #here I reorganized text into lines of single sentence
        for idx in range(len(sentences)):
            text = sentences[idx]
            if os.name == 'nt':
                text = text.replace('\r\n',' ').replace('\n',' ').replace('-',' ')
            else:
                text = text.replace('\n',' ').replace('-',' ')
            sentences[idx] = text
    
    text = ''.join(sentences)
    text = text.replace('  ',' ')
    
    oModel = LexicalSurprisal.CGPT2()
    words, vectors = oModel.get(text)
    
    oStim = CWordStim()
    oStim.loadWordTiming('phonemes1.txt')
    oStim.lowerWords()
    oStim.alignVecToWordTiming(words, vectors)
    impulses = oStim.toImpulses(f = 64)
    return impulses, vectors


# impulses, vectors1 = test_dissimilarity_impulses('glove-wiki-gigaword-100')
impulses, vectors2 = test_dissimilarity_impulses('word2vec-google-news-300')
