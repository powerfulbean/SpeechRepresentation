# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 11:09:58 2022

@author: Jin Dou
@twitter: @jindou1024
"""
from scipy.stats import pearsonr
from scipy.io import loadmat
import numpy as np
import gensim.downloader
import gensim
import os

from ..Utils import lowerWords

def calCtxR(ctxVecList,vec):
    ctxAvgVec = np.mean(ctxVecList,axis=0)
    return pearsonr(ctxAvgVec,vec)[0]

def removeNoneFromList(inList):
    outList = list()
    for i in inList:
        if i is not None:
            outList.append(i)
    return outList

def vecToContextDisVec(vectors,sentDict,debug = False):
    output = np.zeros((len(vectors),1))
    curPnt = 0
    newVecList = [vectors[sentDict[key][0]:sentDict[key][1]] for key in range(len(sentDict))]
    for idx1,sentVec in enumerate(newVecList):
        for idx2,wordVec in enumerate(sentVec):
            try:
                if wordVec is not None:
                    if idx1 == 0 and idx2 == 0:
                        output[curPnt,0] = 0
                    elif idx2 == 0:
                        # calculate pearson with the average of last sentence
                        ctxVecList = newVecList[idx1 - 1]
                        ctxVecList = removeNoneFromList(ctxVecList)
                        if len(ctxVecList) == 0:
                            output[curPnt,0] = 0
                        else:
                            output[curPnt,0] = 1 - calCtxR(ctxVecList, wordVec)
                        pass
                    else:
                        # print(wordVec)
                        # calculate pearson with the average of preceding words in current sentence
                        ctxVecList = sentVec[0:idx2]
                        ctxVecList = removeNoneFromList(ctxVecList)
                        # try:
                        if len(ctxVecList) == 0:
                            output[curPnt,0] = 0
                        else:
                            output[curPnt,0] = 1 - calCtxR(ctxVecList, wordVec)
                        # except:
                            # print(ctxVecList, wordVec)
                        pass
                else:
                    if debug:
                        output[curPnt,0] = -222
            except:
                print(idx1,idx2)
                raise
            curPnt+=1
    return output

def buildDisVec(wordVec,funcWords):
    for idx,i in enumerate(wordVec['word']):
        if i in funcWords:
           wordVec['vec'][idx] = None
    wordVec['vec'] = vecToContextDisVec(
        wordVec['vec'],wordVec['sentence'])
    return wordVec

def processSingleSent(model,tokenizer,sentText):
    sentListWord = list()
    sentListVec = list()
    for token in tokenizer(sentText):
        sentListWord.append(token)
        try:
            vec = model[token]
            sentListVec.append(vec)
        except:
            sentListVec.append(None)
    return sentListWord,sentListVec


class CDissimilarityVector:
    
    def __init__(self, modelName = 'glove-wiki-gigaword-100'):
        print('loading the word2vec')
        self.model = gensim.downloader.load(modelName)
        self.tokenizer = gensim.utils.tokenize
        self.modelName = modelName
        self.funcWordsPath = os.path.join(os.path.dirname(__file__), "funcWords.mat")

    def get(self, sentences):
        singleOutWord = list()
        singleOutVec = list()
        sentDict = dict()
        # cacheWordList = []
        # cacheVecList = []
        for idx,sent in enumerate(sentences):
            hIdx = len(singleOutWord)
            wordList,vecList = processSingleSent(self.model, self.tokenizer,sent)
            singleOutWord += wordList
            singleOutVec += vecList
            tIDx = len(singleOutWord)
            sentDict[idx] = (hIdx,tIDx) #tIdx is not included in this sentence
        
        wordVec = {'word':singleOutWord,'vec':singleOutVec,
                                'sentence':sentDict}
        
        #load michael's funcitonal word
        print(self.funcWordsPath)
        funcWords = [str(i[0][0]) for i in loadmat(self.funcWordsPath)['funcWords']]
        lowerWords(wordVec['word'])
        wordVec = buildDisVec(wordVec,funcWords)
        vectors = wordVec['vec']
        words = wordVec['word']
        return words, vectors
