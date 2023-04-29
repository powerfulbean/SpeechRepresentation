# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 01:09:59 2023

@author: Jin Dou
"""

from transformers import GPT2Tokenizer,GPT2LMHeadModel
import torch
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import (
    CausalLMOutputWithCrossAttentions,
)
import numpy as np
import re

class GPT2LMHeadModelNoReduceLoss(GPT2LMHeadModel):
    '''
    #except the configuration of cross_entropy_loss, and labelsNoShift
    #this function is adopted from GPT2LMHeadModel
    https://github.com/huggingface/transformers/blob/v4.17.0/src/transformers/models/gpt2/modeling_gpt2.py#L1015
    '''
    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labelsNoShift=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            ``labels = input_ids`` Indices are selected in ``[-100, 0, ..., config.vocab_size]`` All labels set to
            ``-100`` are ignored (masked), the loss is only computed for labels in ``[0, ..., config.vocab_size]``
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labelsNoShift is not None:
            # Shift so that tokens < n predict n
            logits = lm_logits[..., :, :].contiguous()
            labels = labelsNoShift[..., 0:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(reduction='none')
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

class CPastKeyValuesCache:
    
    #a Circular queue
    def __init__(self,config):
        #past_key_values:
        # ( ( [nBatch,12,nSeq,64] , [nBatch,12,nSeq,64] ), 
        #     ..., 
        #  ( [nBatch,12,nSeq,64] , [nBatch,12,nSeq,64] )) 
        self.nHead = config.n_head
        self.nPositions = config. n_positions
        self.nEmbed = config.n_embd
        self.nSubEmbed = self.nEmbed // self.nHead
        self._cache = None
    
    def clear(self):
        self._cache = None
    
    def iterKeyValue(self,pastKeyValues):
        for idx,pair in enumerate(pastKeyValues):
            yield pair[0],pair[1]
    
    def iterKeyValuePairBoth(self,pastKeyValues):
        for idx,pair in enumerate(pastKeyValues):
            yield self.data[idx][0],self.data[idx][1],pair[0],pair[1]
    
    def detach(self,pastKeyValues):
        curCache = tuple()
        for key, value in self.iterKeyValue(pastKeyValues):
            curPair = tuple()
            curPair += (key.detach(),)
            curPair += (value.detach(),)
            curCache += (curPair,)
        return curCache
    
    def append(self,pastKeyValues, nLeastEmptySpace = 0):
        pastKeyValues = self.detach(pastKeyValues)
        nSeq = pastKeyValues[0][0].shape[2] #get first layer's first tuple
        if nSeq > self.nPositions - nLeastEmptySpace:
            nElim = nSeq - self.nPositions + nLeastEmptySpace
            curCache = tuple()
            for key, value in self.iterKeyValue(pastKeyValues):
                curPair = tuple()
                curPair += (key[:,:,nElim:,:],)
                curPair += (value[:,:,nElim:,:],)
                curCache += (curPair,)
            self._cache = curCache
        else:
            self._cache = pastKeyValues
        
    @property
    def data(self):
        return self._cache
    
    @property
    def nEmptySpace(self):
        return self.nPositions - self.data[0][0].shape[2] if self.data is not None \
                else self.nPositions

class CInputsQueue:
    '''
    A queue for storing inputs, only pop not append
    '''
    
    def __init__(self,tokens):
        del tokens['attention_mask']
        self._tokens = tokens
        self.hIdx = 0
        self.nTokens = self.data['input_ids'].size()[1]
        temp = self.data['input_ids'][:,1:]
        pad = torch.zeros((temp.shape[0],1))
        pad[:,0] = -100
        pad = pad.long()
        self.labels = torch.cat([temp,pad],dim=1)

    @property
    def data(self):
        return self._tokens
    
    @property
    def nUnaccess(self):
        return self.nTokens - self.hIdx
    
    def pop(self,size,ifLabels = True):
        if self.nUnaccess == 0:
            return None
        elif size > self.nUnaccess:
            size = self.nUnaccess
        else:
            pass
        output = dict.fromkeys(self.data.keys())
        for key in output:
            output[key] = self.data[key][:,self.hIdx:self.hIdx + size]
        
        if ifLabels:
            output['labelsNoShift'] = self.labels[:,self.hIdx:self.hIdx + size]
        self.hIdx += size
        return output
    

class CGPT2:
    
    #surprisal = crossentropyloss in this case
    
    def __init__(self,modelName = 'gpt2',hopSize = 1,useCache = True,ifIncludePuncValue = True):
        self.tokenizer = GPT2Tokenizer.from_pretrained(modelName)
        self.model = GPT2LMHeadModelNoReduceLoss.from_pretrained(modelName).eval()
        self.hopSize = hopSize
        self.specialTokens = self.tokenizer.all_special_tokens
        self.SPACETOKEN = 'Ġ'
        self.puncSet = " .,?!';\")("
        self.useCache = useCache
        self.nPositions = self.model.config.n_positions
        self.hopSize = hopSize
        self.oPastKeyValue = CPastKeyValuesCache(self.model.config)
        self.ifIncludePuncValue = ifIncludePuncValue

    def fValue(self,token,value):
        if not self.ifIncludePuncValue:
            if any([token==i for i in self.puncSet]):
                value = 0
        return value

    def get(self,text,reduce = 'sum'):
        wordList,vecList = self.textToVec(text, self.getSurprisal)
        wordList,vecList = self.textToVecPostProcess(wordList,vecList,reduce = reduce)
        return wordList, np.array(vecList)
        
    def getSurprisal(self,output,idx,tokens):
        loss = output['loss'][idx].item()
        return np.array([loss])
    
    def textToVec(self,sent,fVecProcess,seperateQuote = True,inputsOperation = lambda x:x,clearMemForEachSent = True,ifIncludePuncValue = False):
        self.ifIncludePuncValue = ifIncludePuncValue
        assert callable(fVecProcess)
        sentListWord = list()
        sentListVec = list()
        vecQueue = list()
        #append 0 for the first token, since no previous token for it
        vecQueue.append(np.array([0])) 
        sentText = sent
        inputsTotal = self.tokenizer(sentText, return_tensors="pt")
        inputsTotal = inputsOperation(inputsTotal)
        oInput = CInputsQueue(inputsTotal)
        
        cnt = 0
        # for inputs in self.fetchInput(inputsTotal):
        while oInput.nUnaccess > 0:
            inputs = oInput.pop(self.oPastKeyValue.nEmptySpace)
            inputs['use_cache'] = self.useCache
            inputs['past_key_values'] = self.oPastKeyValue.data
            self.input = inputs
            #need more logics at here, for prediction
            outputs = self.model(**inputs)
            self.oPastKeyValue.append(outputs.get('past_key_values'),nLeastEmptySpace=self.hopSize)
            inputIds = inputs['input_ids']
            tokens = self.tokenizer.convert_ids_to_tokens(inputIds[0].numpy())
            for idx,i in enumerate(tokens):
                # print(i)
                print('\r converting data: idx:{}'.format(cnt),end='\r')
                cnt += 1
                # print(f'converting token: {i}')
                # if all([sp not in i for sp in self.specialTokens]):
                if i not in self.specialTokens:
                    vecQueue.append(fVecProcess(outputs,idx,inputs))
                    value = vecQueue.pop(0)
                    if i == self.SPACETOKEN:
                        if idx < len(tokens) - 1:
                            if tokens[idx+1][0] != self.SPACETOKEN:
                                tokens[idx+1] = self.SPACETOKEN + tokens[idx+1]
                    elif self.SPACETOKEN in i or idx == 0:
                        sentListWord.append([i.replace(self.SPACETOKEN,'')])
                        sentListVec.append([self.fValue(i,value)])
                    else:
                        if seperateQuote and "'" in i:
                            if i[-1] == "'":
                                sentListWord[-1].append(i[:-1])
                                sentListVec[-1].append(self.fValue(i,value))
                                tokens[idx + 1] = self.SPACETOKEN + tokens[idx + 1]
                            elif i[0] == "'":
                                sentListWord.append([i[1:]])
                                sentListVec.append([self.fValue(i,value)])
                            else:
                                raise NotImplementedError
                        else:
                            sentListWord[-1].append(i)
                            sentListVec[-1].append(self.fValue(i,value))

        
        if clearMemForEachSent:
            self.oPastKeyValue.clear()
        
        return sentListWord,sentListVec
    
    def textToVecPostProcess(self,sentListWord,sentListVec, in_place = True, 
                             keepPunc = False, lower = True, reduce = 'mean'):
        '''
        Note that this opetation is in_place by default
        '''
        if reduce == 'mean':
            fReduce = np.mean
        elif reduce == 'sum':
            fReduce = np.sum
        else:
            raise ValueError
        if not in_place:
            sentListWord = sentListWord.copy()
            sentListVec  = sentListVec.copy()
        #post process
        for idx, i in enumerate(sentListWord):
            # print(i)
            if len(i) == 1:
                if lower:
                    sentListWord[idx] = i[0].lower()
                else:
                    sentListWord[idx] = i[0]
                sentListVec[idx]  = sentListVec[idx][0]
            elif len(i) == 0:
                raise ValueError
            else:
                if keepPunc:
                    sentListWord[idx] = "".join(i)
                    sentListVec[idx]  = fReduce(sentListVec[idx],axis = 0)
                else:
                    # sentListWord[idx] = "".join(i).lower().strip(" .,?!';\":[]")
                    temp = "".join(i).lower()
                    sentListWord[idx] = re.sub('[.!,?\]\[\":;\)\(]','',temp).strip("'") # \'
                    # if "." in sentListWord[idx]:
                        # sentListWord[idx] = sentListWord[idx].replace('.','')
                    sentListVec[idx]  = fReduce(sentListVec[idx],axis = 0)
        
        for idx in range(len(sentListWord)-1,-1,-1):
            if len(re.sub('[.!,?\]\[\'\":;\)\(]','',sentListWord[idx])) == 0:
                if idx > 0:
                    sentListVec[idx-1] = (sentListVec[idx-1] + sentListVec[idx]) / 2
                del sentListWord[idx]
                del sentListVec[idx]
            
        
        return sentListWord,sentListVec
