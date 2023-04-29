# -*- coding: utf-8 -*-
import os
from SpeechRepresentation.Semantic import SemanticDissimilarity, LexicalSurprisal
from SpeechRepresentation.Utils import CWordStim


def test_dissimilarity_impulses():
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
    
    oModel = SemanticDissimilarity.CDissimilarityVector()
    words, vectors = oModel.get(sentences)
    
    oStim = CWordStim()
    oStim.loadWordTiming('phonemes1.txt')
    oStim.lowerWords()
    oStim.alignVecToWordTiming(words, vectors)
    impulses = oStim.toImpulses(f = 64)
    return impulses

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
    return impulses