import sys
import nltk
from numpy import *
import numpy as np
from nltk.util import ngrams

##get files from command line
testfile = (sys.argv[2]) + ".txt"
trainfile = (sys.argv[1]) + ".txt"
##getting transition and emission states/processing training data
traindata = open(trainfile, "r")
transitions= []
countline = 0
emissions = []
words = []
for line in traindata:
    countline+=1
    line = line.replace("\n", "") 
    arr = line.split("/")
    transitions.append(arr[1])
    emissions.append(arr)
    words.append(arr[0])
transitiongram = list(ngrams(transitions,2))
n = len(words)
V = len(set(words))
##generating transition and emission probabilities
transitionprob = {}
#get unique tags from training data
transitionset = []
for tag in transitions:
    if tag not in transitionset:
        transitionset.append(tag)
if '#' in transitionset :
    transitionset.remove('#')
#tag to tag probabilities
for rowitem in transitionset:
    for colitem in transitionset:
        transitioncount = 0
        for item in transitiongram:
            if (rowitem, colitem) == item :
                transitioncount+=1
        #using formula given in smoothing document
        transitionprob[(rowitem,colitem)] = transitioncount+1/(n+(V+1))
transitionprob['UNK'] = 1/(n+(V + 1))

#get unique emissions (words)
tagdict = {}
emissionprob = {}
emissionset = set(words)
emissionset = (sorted(emissionset))
for item in emissionset:
    if '#' in emissionset :
        emissionset.remove(item)
#get word to tag probabilites and generate tag dict
for rowitem in emissionset:
    arr = []
    arr1 = []
    for colitem in transitionset:
        emissioncount = 0
        for item in emissions:
            if [rowitem, colitem] == item :
                if colitem not in arr :
                    arr.append(colitem)
                emissioncount+=1
        tagdict[rowitem] = arr
        #using formula given in smoothing document
        emissionprob[(rowitem,colitem)] = emissioncount + 1/(n + (V+1))
emissionprob['UNK'] = 1/(n + (V+1))
tagdict['UNK'] = transitionset

##get test data and process test data
testdata = open(testfile, "r")
testwords = []
actualtags = []
countline = 0
for line in testdata:
    countline+=1
    line = line.replace("\n", "") 
    arr = line.split("/")
    testwords.append(arr[0])
    actualtags.append(arr[1])
actualtags.remove('#')
testwords.remove('#')
testsize = len(testwords)

##Viterbi tagger

#initialize trellis and backtrace
trellis = np.zeros((testsize , len(transitionset)))
backtrace = np.empty((testsize -1 , len(transitionset)), dtype=object)
trellis[0,0] = 1
known = True
knowncount = 0
unknowncount = 0
current = []
previous = []
for i in range (0, testsize) :
    if testwords[i] in words:
        tag1 = tagdict.get(testwords[i])
    else :
        tag1 = tagdict.get('UNK')
    #go through current tag set
    for j in range (0, len(tag1)) :
        currenttag = tag1[j]
        current.append(currenttag)
        #known word
        if testwords[i-1] in words:
            nexttag = tagdict.get(testwords[i-1])
        #unknown word
        else :
            nexttag = tagdict.get('UNK')
        #go through previous tag set
        for k in range (0, len(nexttag)) :
            previoustag = nexttag[k]
            previous.append(previoustag)
            transitiongram = (actualtags[i-1], actualtags[i])
            emissiongram = (testwords[i], currenttag)
            #generate probability (using formula given in psuedocode)
            if transitiongram in transitionprob and emissiongram in emissionprob :
                known = True
                probability = transitionprob.get(transitiongram) * emissionprob.get(emissiongram)
            else :
                known = False
                transitiongram = ('UNK')
                emissiongram = ('UNK')
                probability = transitionprob.get(transitiongram) * emissionprob.get(emissiongram)
            variableu = trellis[i-1, k] * probability
            #add to trellis and backtrace if needed
            if variableu >= trellis[i,j] :
                trellis[i,j] = variableu
                backtrace[i - 1 ,j] = [previoustag, known]
backtrace[testsize - 2, len(transitionset) - 2] = '###'
##get sequence & tags (tags included known/unknown)
sequence = []
tags = []
for i in range (0,testsize-1) :
    for element in backtrace[i] :
        if element != None :
            sequence.append(element[0])
            break
    tags.append(element)

##calculate overall accuracy
accuratetags = 0
for i in range (0,len(actualtags) - 1) :
    if actualtags[i] == sequence[i] :
            accuratetags+=1
print("Total accurate tags:")
print(accuratetags)

#calculate known and unknowns
totalknown = 0
totalunknown = 0
knowncorrect = 0
unknowncorrect = 0
for i in range (0, len(actualtags) - 1) :
    arr = tags[i]
    #known word accuracy
    if arr[1] == True :
        totalknown+=1
        if arr[0] == actualtags[i] :
            knowncorrect+=1
    #unknown word accuracy
    else :
        totalunknown+=1
        if arr[0] == actualtags[i] :
            unknowncorrect+=1
print("Total known & known correct:")
print(totalknown)
print(knowncorrect)
print("Total unknown & unknown correct:")
print(totalunknown)
print(unknowncorrect)
        
    