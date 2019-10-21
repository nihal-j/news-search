#!/usr/bin/env python
# coding: utf-8

# Done so far :
# 
# 
# *   Lemmatization
# *   Stop Words Removal
# 
# Verify :
# 
# * Normalization - removing accents, etc.
# * Dates replaced with strings
# * Case-folding
# * Removed HTML entity codes
# 
# 

# In[1]:


import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import re
import wordninja 

####### After importing nltk, run the following only once ######
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')
# nltk.download('stopwords')
### pip install wordninja ###


# In[2]:


def remove_htmlcodes(document):
    
    '''Removes HTML entity codes such as &amp from document and returns the clean document'''
    
    replacement = {
                    "&ampnbsp": ' ',
                    "&ampamp": '&',
                    "&ampquot": '\'',
                    "&ampldquo": '\"',
                    "&amprdquo": '\"',
                    "&amplsquo": '\'',
                    "&amprsquo": '\'',
                    "&amphellip": '...',
                    "&ampndash": '-',
                    "&ampmdash": '-'
                  }
    
    for str in replacement:
        document = document.replace(str, replacement[str])
        
    return document


# In[3]:


def get_wordnet_pos(word):
    
    '''Returns the tag of usage of word depending on context'''
    
    tag=nltk.pos_tag([word])[0][1][0].upper()
    tag_dict={"J": wordnet.ADJ, 
              "N": wordnet.NOUN,
              "V": wordnet.VERB,
              "R": wordnet.ADV}
    return tag_dict.get(tag,wordnet.NOUN)

def lemma_stop(str):
    
    lemmatizer = WordNetLemmatizer()
    tokenizer = RegexpTokenizer('\w+|\$]\d\[+|\S+,-')
    tokenized = tokenizer.tokenize(str)
    lemmatized = [lemmatizer.lemmatize(w,get_wordnet_pos(w)) for w in tokenized]
    stop_words = set(stopwords.words('english'))
    filtered_sentence = [w for w in lemmatized if w.lower() not in stop_words]
    after_lemma_stop = ' '.join(w for w in filtered_sentence)
    return filtered_sentence


# In[4]:


def is_not_credible (text):
    
    '''Returns true if text has no special characters, else returns false'''
    
    match = re.search(r'[!@#?&{}()]', text)
    
    if match:
        return True
    else:
        return False


# In[5]:


def scrub_words(text):
    
    '''Removes special characters from text and returns a clean string'''
    
    text = re.sub('[!@#?&{}()]', '', text)
    text=re.sub(r'[^\x00-\x7F]'," ",text)
    return text


# In[6]:


def clean_document (document_string):
    
    '''Cleans document_string by splitting very long strings and identifying garbage JSON and HTML and discarding'''

    
    cleaned_doc = document_string
    for word in document_string.split():
                if is_not_credible(word):
                    temp= scrub_words(word)
                    split=wordninja.split(temp)
                    if len(split)>7:
                          cleaned_doc = cleaned_doc.replace(word,'')
                    else:
                        replace_with=' '.join(word for word in split)
                        cleaned_doc = cleaned_doc.replace(word, replace_with)
    return cleaned_doc


# In[7]:


def replace_dates(documentString, docID):
    
    '''Replaces dates of the format MM/DD and MM/DD/YYYY with DDmmmYYYY inside documentString'''
    
    from datetime import datetime
    count_dates = []
    
    regEx = '(([0-9]+(/)[0-9]+(/)[0-9]+)|([0-9]+(/)[0-9]+))'
    iterator = re.finditer(regEx, documentString)
    listOfDates = [(m.start(0), m.end(0)) for m in iterator]
    tmp = []
    replace_with = []
    for indices in listOfDates:
        date = documentString[indices[0]:indices[1]]
        tmp.append(date)
        count = date.count('/')
        newDate = ''
        if count == 2:
            check_year = date[-3]
            
            if check_year == '/':
                YY = date[-2:]
                
                if int(YY) <= 19:
                    proper_date = date[:-2] + '20' + YY
                    date = date.replace(date,proper_date)
                else:
                    proper_date = date[:-2] + '19' + YY
                    date = date.replace(YY,('19'+YY))
                    
            try:
                newDate = datetime.strptime(date, '%m/%d/%Y').strftime('%d %b %Y')
            except ValueError as ve:
                newDate = date
        else:
            try:
                newDate = datetime.strptime(date, '%m/%d').strftime('%d %b')
            except ValueError as ve:
                newDate = date
                
        count_dates.append([docID, date])
        newDate = newDate.replace(' ', '')
        replace_with.append(newDate)
        
    for i in range(len(tmp)):
        documentString = documentString.replace(tmp[i], replace_with[i])
    
    return documentString


# In[14]:


################################################
## ------------ PREPROCESSING --------------- ##
##              run only once                 ##
################################################

def preprocess(data):
    
    '''
    Performs the following on data:
        1. Lemmatization
        2. Stop Words Removal
        3. Normalization - removing accents, etc.
        4. Replacing dates with strings
        5. Case-folding
        6. Removed HTML entity codes
    '''
    
    import time
    from tqdm import tqdm_notebook
    
    start = time.time()

    titles = []
    contents = []
    lower = len(data) // 2
    upper = lower + 3000

    for i in tqdm_notebook(range(lower, upper)):

        if data[i][4] == None or data[i][1] == None or data[i][0] == None:
            continue

        # actually modifying the document
        data[i][4] = remove_htmlcodes(data[i][4])
        data[i][1] = remove_htmlcodes(data[i][1])
        data[i][4] = clean_document(data[i][4])
        data[i][1] = clean_document(data[i][1])

        # not actually modifying the document
        modifiedContent = replace_dates(data[i][4], data[i][0])
        modifiedContent = lemma_stop((modifiedContent))
        modifiedTitle = replace_dates(data[i][1], data[i][0])
        modifiedTitle = lemma_stop((modifiedTitle))

        # case-folding
        for j in range(len(modifiedContent)):
            modifiedContent[j] = modifiedContent[j].lower()
        for j in range(len(modifiedTitle)):
            modifiedTitle[j] = modifiedTitle[j].lower()

        titles.append(modifiedTitle)
        contents.append(modifiedContent)

    # filet = "/home/nihaljain/3-1/CS F469/Assignment-1/mod_titles"
    # filec = "/home/nihaljain/3-1/CS F469/Assignment-1/mod_contents"
    # filed = "/home/nihaljain/3-1/CS F469/Assignment-1/mod_data"

    # np.save(filet, titles)
    # np.save(filec, contents)
    # np.save(filed, data)

    print(time.time() - start)  # 110.26236414909363
    
    return data, contents, titles

    # --------------------OPTIONALLY------------------------

    # contents = []
    # titles = []
    # data = []

    # filet = "/home/nihaljain/3-1/CS F469/Assignment-1/mod_titles"
    # filec = "/home/nihaljain/3-1/CS F469/Assignment-1/mod_contents"
    # filed = "/home/nihaljain/3-1/CS F469/Assignment-1/mod_data"

    # titles = np.load(filet + ".npy", allow_pickle = True)
    # contents = np.load(filec + ".npy", allow_pickle = True)
    # data = np.load(filed + ".npy", allow_pickle = True)


# In[11]:


#-----------NOTE------------
# len(contents) != len(data) // 2

def remove_accents(contents, titles):
    
    '''Removes accents from all strings in contents and titles'''
    
    import unidecode
    import pickle

    for i in range(len(contents)):
        for j in range(len(contents[i])):
            contents[i][j] = unidecode.unidecode(contents[i][j])
        for j in range(len(titles[i])):
            titles[i][j] = unidecode.unidecode(titles[i][j])
    
    # OPTIONALLY TO SAVE PERSISTENTLY #
    
    # with open('modified_contents_ascii.pickle', 'wb') as handle:
    #    pickle.dump(contents, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # with open('modified_titles_ascii.pickle', 'wb') as handle:
    #    pickle.dump(titles, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    return contents, titles


# In[12]:


def construct_corpus(contents, titles, data):
    
    '''Constructs global corpus trie and several document tries using contents, titles and data'''

    import pickle
    import time
    from tqdm import tqdm_notebook

    max_tf = {}
    start = time.time()
    j = 0
    lower = len(data) // 2
    upper = len(data) // 2 + 3000


    for i in tqdm_notebook(range(lower, upper)):

        if data[i][4] == None or data[i][1] == None or data[i][0] == None:
            continue

        for w in contents[j]:
            collection.add_document(w, 0, get_docID[i])
            documentRoot[get_docID[i]].add(w, 0)

            if get_docID[i] in max_tf:
                max_tf[get_docID[i]] = max(documentRoot[get_docID[i]].count_words(w, 0), max_tf[get_docID[i]])
            else:
                max_tf[get_docID[i]] = documentRoot[get_docID[i]].count_words(w, 0)

        for w in titles[j]:
            collection.add_title(w, 0, get_docID[i])

        j += 1

    print(time.time() - start)  #39.19705152511597

    return collection, documentRoot, max_tf


# In[15]:


if __name__ == '__main__':
    
    import trie
    import pickle
    
    # loading data.npy
    # data.npy is a 2D array containing the dataset information as
    # data[i][0] : docID of ith document
    # data[i][1] : title of ith document
    # data[i][4] : content of ith document
    data = np.load('data.npy',allow_pickle = True)
    
    # preprocessing
    data, contents, titles = preprocess(data)
    contents, titles = remove_accents(contents, titles)  
    
    # print(len(data), len(contents), len(titles))
    
    # constructing the tries
    
    getReference = {}
    get_docID = {}
    get_index = {}

    for i in range(0, len(data)) :
        get_docID[i] = int(data[i][0])
        get_index[int(data[i][0])] = i
        
    documentRoot = {}
    collection = trie.CollectionNode()

    # initializing the root for N documents
    
    lower = len(data) // 2
    upper = lower + 3000
    
    for i in range(lower, upper):
        newDocument = trie.Node()
        documentRoot[get_docID[i]] = newDocument
        
    collection, documentRoot, max_tf = construct_corpus(contents, titles, data)
    
    # saving to pickle files
    
    # with open('collection.pickle', 'wb') as handle:
    #    pickle.dump(collection, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # with open('documentRoot.pickle', 'wb') as handle:
    #    pickle.dump(documentRoot, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # with open('max_tf.pickle', 'wb') as handle:
    #    pickle.dump(max_tf, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # reading from pickle files

    # with open('collection.pickle', 'rb') as handle:
    #     collection = pickle.load(handle)
    # with open('documentRoot.pickle', 'rb') as handle:
    #     documentRoot = pickle.load(handle)


# In[49]:


# import math
# import queue

# documentLength = {}
# N = len(documentRoot)

# for i in tqdm(range(len(documentRoot))):
    
#     docID = get_docID[i]
#     length = 0
#     document = documentRoot[i]
#     q = queue.Queue()
#     q.put([document, ''])

#     while q.qsize() > 0:

#         current = q.get()
#         reference = current[0]
#         word = current[1]

#         if reference.words > 0:
#             df = len(collection.get_doc_list(word, 0))
#             idf = math.log10(N/df)
#             # print(word, reference.words, df)
#             length += (reference.words * idf) ** 2

#         for i in range(256):
#             if reference.children[i] is not None:
#                 new_word = word + chr(i)
#                 q.put([reference.children[i], new_word])

#     # print(length**0.5)
#     documentLength[docID] = length**0.5

