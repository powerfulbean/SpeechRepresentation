# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 23:49:40 2023

@author: Jin Dou
"""
import warnings
import pandas as pd
import numpy as np

def lowerWords(words):
    for i,word in enumerate(words):
        words[i] = word.lower()
    return words

class CWordStim:
    
    def __init__(self,):
        self.words = []
        self.startTimes = []
        self.endTimes = []
        self.vectors = []
        
    def loadWordTiming(self,filePath):
        dataframe = pd.read_csv(filePath)
        words = dataframe['word'].tolist()
        startTimes =  dataframe['startTime'].tolist()
        endTimes =  dataframe['endTime'].tolist()
        self.words.extend(words)
        self.startTimes.extend(startTimes)
        self.endTimes.extend(endTimes)
        self.vectors.extend([None] * len(words))
        return words, startTimes, endTimes
    
    def alignVecToWordTiming(self, words, vectors, removeEdgeUnmatched = True, fReduce = np.mean):
        #this function is necessary in case the words from word timing related files 
        #   are not exactly the same as the words returned by NLP models
        idx = 0
        for idx2, i in enumerate(self.words):
            if idx >= len(words):
                maxIdx = len(self.words)
                for idxDel in range(maxIdx,idx2,-1):
                    self.vectors.pop(-1)
                    self.startTimes.pop(-1)
                    self.endTimes.pop(-1)
                    self.words.pop(-1)
                break
            if i.replace('.','') == words[idx].replace('.','') or words[idx] == '<unk>':
                self.vectors[idx2] = vectors[idx]
                idx += 1
            elif words[idx] in i:
                vecListTemp = [vectors[idx]]
                charListTemp = [words[idx]]
                idx += 1
                while ''.join(charListTemp) in i and ''.join(charListTemp) != i:
                    vecListTemp.append(vectors[idx])
                    charListTemp.append(words[idx])
                    idx += 1
                self.vectors[idx2] = fReduce(vecListTemp,axis=0)
            else:
                raise ValueError(idx2,words[idx],i)
        if len(words) < len(self.words):
            print(len(words),len(self.words))
        assert len(words) >= len(self.words)
        self.vectors = np.array(self.vectors)
        self.startTimes = np.array(self.startTimes)
        self.endTimes = np.array(self.endTimes)
    
    def toImpulses(self,f:float,padding_s = 0):
        '''
        # align the vectors into impulses with specific sampling rate 
        '''
        secLen = self.endTimes[-1] + padding_s
        nLen = np.ceil( secLen * f).astype(int)
        nDim = self.vectors.shape[1]
        out = np.zeros((nLen,nDim))
        
        timeIndices = np.round(self.startTimes * f).astype(int)
        out[timeIndices,:] = self.vectors
        return out
    
    def lowerWords(self):
        lowerWords(self.words)

# class CWordVecLabel:
    
#     class CLabelKeyAdapter:
#         '''
#         An adapter is a class for transferring a dictionary's keywords to the 
#             keywords required by WordVecLabel
#         '''
#         def __init__(self,word, startTime, endTime):
#             self._map = {WORD:word,START_TIME:startTime,END_TIME:endTime}
        
#         def __call__(self,inputDict:dict):
#             out = dict()
#             for i in self._map:
#                 out[i] = inputDict[self._map[i]]
#             return out
        
#     def loadFile(self,file):
#         dataframe = pd.read_csv(file)
#         return self.packDF(dataframe)
    
#     def packDF(self, matData) -> dict:
#         out = dict()
#         out['word'] = matData['word'].tolist()
#         out['startTime'] =  matData['startTime'].tolist()
#         out['endTime'] =  matData['endTime'].tolist()
#         return out #if use dict directly, will be treated as series
    
#     def __init__(self,dataDict=None,nFeat = 1):
#         super().__init__(nFeat)
#         self.key = -1
#         if dataDict is not None:
#             self.readFile(dataDict)
        
#     def readFile(self,dataDict):
        
#         if isinstance(dataDict,str):
#             dataDict = self.loadFile(dataDict)
        
#         self.timestamps.clear()
#         self.rawdata.clear()
        
#         word = dataDict[WORD]
#         startTime = dataDict[START_TIME]
#         endTime = dataDict[END_TIME]
        
#         for i in range(len(word)):
#             s = startTime[i]
#             e = endTime[i]
#             w = word[i].lower()
#             oW = CWordStimulus(self.nFeat)
#             oW.word = w
#             self.append(CTimeIntervalStamp(s,e), oW)
           
#     def loadStimuli(self,word,vec,preventDuplicate = True):
#         #load  vector for each word
#         out = self.rawdata.copy()
#         if preventDuplicate:
#             if self.rawdata[:,-2].word.lower() == word[-1].lower():
#                 wordTemp = self.rawdata[:,-1].word
#                 del self.rawdata._list[-1]
#                 del self.timestamps[-1]
#                 warnings.warn(f'the last {wordTemp} is removed')
#             if self.rawdata[:,1].word.lower() == word[0].lower():
#                 wordTemp = self.rawdata[:,0].word
#                 del self.rawdata._list[0]
#                 del self.timestamps[0]
#                 warnings.warn(f'the first {wordTemp} is removed')
#         idx = 0
#         for idx2, i in enumerate(self.rawdata):
#             # print(idx2,idx,i.word.replace('.',''), word[idx].replace('.',''),i.word.replace('.','') == word[idx].replace('.',''))
#             if idx >= len(word):
#                 maxIdx = self.rawdata.shape[1]
#                 for idxDel in range(maxIdx,idx2,-1):
#                     self.rawdata._list.pop(-1)
#                     self.timestamps.pop(-1)
#                 break
#             if i.word.replace('.','') == word[idx].replace('.','') or word[idx] == '<unk>':
#                 self.rawdata[:,idx2] = vec[idx]
#                 idx += 1
#             elif word[idx] in i.word:
#                 vecListTemp = [vec[idx]]
#                 charListTemp = [word[idx]]
#                 idx += 1
#                 while ''.join(charListTemp) in i.word and ''.join(charListTemp) != i.word:
#                     vecListTemp.append(vec[idx])
#                     charListTemp.append(word[idx])
#                     idx += 1
#                 self.rawdata[:,idx2] = np.mean(vecListTemp,axis=0)
#             elif i.word in word[idx]: #when the wordOnset 
#                 while self.rawdata(idx2).word in word[idx] and self.rawdata(idx2).word != word[idx]:
#                     self.rawdata._list[idx2] = self.rawdata(idx2).add(self.rawdata(idx2+1))
#                     self.rawdata._list.pop(idx2+1)
#                     self.timestamps[idx2].endTime = self.timestamps[idx2+1].endTime
#                     self.timestamps.pop(idx2+1)
#                 self.rawdata[:,idx2] = vec[idx]
#                 idx += 1
#             else:
#                 raise ValueError(idx2,word[idx],i.word)
#         if len(word) < self.rawdata.shape[1]:
#             print(len(word),self.rawdata.shape[1])
#         assert len(word) >= self.rawdata.shape[1]
#         return out
    
#     def lowerWord(self):
#         for i in self.data:
#             i.word = i.word.lower()
            
#     def cleanPunc(self,ifEdgeOnly = False):
#         if ifEdgeOnly:
#             for i in self.data:
#                 i.word = i.word.strip('[.!,?\]\'\[\":;\-\(\)]')
#         else:
#             for i in self.data:
#                 i.word = re.sub('[.!,?\]\'\[\":;\-\(\)]','',i.word)
#     # def loadStimuli(self,word,vec):
#     #     #load  vector for each word, upgraded for being more flexible
#     #     word = word.copy()
#     #     vec = vec.copy()
#     #     out = self.rawdata.copy()
#     #     # idx = 0
#     #     curWord = word.pop(0)
#     #     curVec = vec.pop(0)
#     #     for idx2, i in enumerate(self.rawdata):
#     #         if i.word.replace('.','') == curWord.replace('.','') or curWord == '<unk>':
#     #             print('?',idx2,i.word,curWord)
#     #             self.rawdata[:,idx2] = curVec#vec[idx]
#     #             curVec = vec.pop(0) if len(vec) > 0 else ''
#     #             curWord = word.pop(0) if len(word) > 0 else ''
#     #             print('?!',idx2,i.word,curWord, len(word) > 0)
#     #             # idx += 1
#     #         elif curWord in i.word:
#     #             print('!',idx2,i.word,curWord)
#     #             self.rawdata[:,idx2] = np.mean([curVec,vec.pop(0)],axis=0) #np.mean([vec[idx],vec[idx+1]],axis=0)
#     #             # idx += 2
#     #             word.pop(0) #because we pop an additional vec
#     #             curVec = vec.pop(0) if len(vec) > 0 else ''
#     #             curWord = word.pop(0) if len(word) > 0 else ''
#     #         else:
#     #             print(idx2,i.word,curWord)
#     #             pass
#     #         if len(vec) == 0:
#     #             assert len(vec) == len(word)
#     #             break
#     #     print(word)
#     #     assert len(word) == 0
#     #     assert len(vec) == 0
#     #     return out
    
#     def combinedSpan(self,f:float,onsetVec = 0.5, ifOnsetOnly = False,padding_s = 0):
#         '''
#         # span the time-interval labeled label into sequence with specific sampling rate 
#         Parameters
#         ----------
#         f : float
#             sampling rate.

#         Returns
#         -------
#         spanned stimuli array

#         '''
#         secLen = self.timestamps[-1].endTime + padding_s
#         nLen = int(np.ceil( secLen * f))
#         nDim = self.data.shape[0]
#         # oldVec = onsetVec
#         if onsetVec is None:
#             out = np.zeros((nDim,nLen))
#             # print(out[:,sec_num].shape)
#         elif isinstance(onsetVec, slice):
#             out = np.zeros((nDim + 1,nLen))
#             labelVecs = self.data[onsetVec]
#             onsetVec = np.mean(labelVecs)
#             # print(onsetVec,labelVecs.shape)
#         elif ifOnsetOnly:
#             out = np.zeros((1,nLen))
#         else:
#             out = np.zeros((nDim + 1,nLen))
#         # print(onsetVec) #oldVec,
#         for idx,i in enumerate(self.timestamps):
#             sec_num = int(np.round(i.startTime * f)) ## ??? ceil or round
#             if onsetVec is None:
#                 # print(a)
#                 out[:,sec_num] = self.data[:,idx]
#                 # print(out[:,sec_num].shape)
#             elif ifOnsetOnly:
#                 out[:,sec_num] = np.array([onsetVec])
#             else:
#                 if self.data[:,idx] == 0:
#                     out[:,sec_num] = np.concatenate([self.data[:,idx],[0]])
#                 else:
#                     out[:,sec_num] = np.concatenate([self.data[:,idx],[onsetVec]])
#         return out