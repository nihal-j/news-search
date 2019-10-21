#!/usr/bin/env python
# coding: utf-8

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


from datetime import datetime

count_dates = []

def replace_dates(documentString, docID):
    
    '''Replaces dates of the format MM/DD and MM/DD/YYYY with DDmmmYYYY inside documentString'''
    
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


# In[34]:


def preprocess_query(query):
    
    '''
    Performs the following on query:
        1. Lemmatization
        2. Stop Words Removal
        3. Normalization - removing accents, etc.
        4. Replacing dates with strings
        5. Case-folding
    '''

    import unidecode

    final_query = replace_dates(query, -1)
    final_query = lemma_stop(final_query)

    for i in range(len(final_query)):
        final_query[i] = unidecode.unidecode(final_query[i])
        # case-folding
        final_query[i] = final_query[i].lower()

    tf_query = {}
    for w in final_query:
        if w not in tf_query:
            tf_query[w] = 1
        else:
            tf_query[w] += 1
    return tf_query, final_query


# ***Ranked Retrieval based on TF-IDF Score :***
# 

# In[35]:


def calculate_scores(collection, documentRoot, max_tf, tf_query, final_query):
    
    '''Prints relevant results to the query using the vector space model'''

    import queue
    import math
    import bisect

    # scores[i] stores the dot product of the tf-idf score vectors of the query and document of docID i in the corpus
    scores = {}
    title_score = {}

    # N is the total number of documents in the corpus
    N = len(documentRoot)

    # wordsInDoc[i] is a sorted list of (word, score) tuples where
    # score is the tf-idf score for the (word, <ith doc>) pair
    wordsInDoc = {}

    factor = {}

    # stores a list of docIDs of the documents presented to the user
    retrieved_docs = []

    for query_term in tf_query:

        docs_having_query_term = collection.get_doc_list(query_term, 0)
        df = len(docs_having_query_term)
        idf = 0

        print('----------------------------------------------------------------------------------------')
        print('Term in query = ', query_term)
        print()

        if df == 0:
            idf = 0
        else:
            idf = math.log10(N/df)

        docs_having_query_term_in_title = collection.get_title_list(query_term,0)

        for docID in docs_having_query_term_in_title:

            if docID in title_score:
                title_score[docID] += idf
            else:
                title_score[docID] = idf

        print('df = ',df)
        print('idf = ',idf)

        tfidf_query = tf_query[query_term] * idf

        for docID in docs_having_query_term:

            tf_doc = documentRoot[docID].count_words(query_term, 0)
            tf_doc = 0.5 + 0.5*tf_doc/max_tf[docID]
            tfidf_doc = (tf_doc)

            if docID not in scores:
                scores[docID] = (tfidf_query * tfidf_doc)
                wordsInDoc[docID] = []
                bisect.insort(wordsInDoc[docID], [-tfidf_query * tfidf_doc, query_term])
                factor[docID] = idf
            else:
                scores[docID] += (tfidf_query * tfidf_doc)
                bisect.insort(wordsInDoc[docID], [-tfidf_query * tfidf_doc, query_term])
                factor[docID] += idf

    # print(title_score)

    for docID in scores:

        #if documentLength[docID] != 0:
        scores[docID] *= factor[docID]
        if docID in title_score:
            scores[docID] *= 1 + title_score[docID]

    sorted_scores = sorted(scores.items(), key = lambda kv : kv[1] , reverse = True)

    maxshow = min(10, len(scores))

    print('\n\n')
    print('========================================================================================')

    for i in range(maxshow):

        print()
        docID = sorted_scores[i][0]
        retrieved_docs.append(docID)
        print('doc ID = ', docID)
        cnt = 0
        print('Keywords:', end = ' ')
        for j in range(len(wordsInDoc[docID])):
            print(wordsInDoc[docID][j][1], end = ' ')
        print()
        print()
        print(data[get_index[sorted_scores[i][0]]][1])
        print()
        
        ## printing information
        if sorted_scores[i][0] not in title_score:
            print('title score = ',0)
        else:
            print('title score = ',title_score[sorted_scores[i][0]])
        for j in range(len(wordsInDoc[docID])):
            print(wordsInDoc[docID][j][1], wordsInDoc[docID][j][0], end = ' ')
            print(documentRoot[docID].count_words(wordsInDoc[docID][j][1], 0))
        print()
        print()
        ## end of information
        
        count = 0
        found = 0
        words_before=queue.Queue()
        at_start = 1
        display = ""

        for word in data[get_index[docID]][4].split():

            check_with=replace_dates(word, -1)
            check_with = check_with.lower()
            if len(lemma_stop(check_with)) > 0:
                check_with=lemma_stop(check_with)[0]
            else:
                check_with=word

            if check_with == wordsInDoc[docID][0][1]:
                found=1

            if found == 1:
                display = display + word + " "
                count += 1
                if count == 50:
                    break
            if found == 0:
                words_before.put(word)
                if words_before.qsize()>20:
                    remove=words_before.get()
                    at_start=0

        if not at_start:
            print('...', end = ' ')
        while words_before.qsize() > 0:
            print(words_before.get(), end = ' ')
        print(display, end = ' ')
        print('...', end = ' ')
        print('\n')
        print('tf-idf score=', sorted_scores[i][1])
        print('\n')
        print('========================================================================================')

    return retrieved_docs

# In[14]:


if __name__ == '__main__':
    
    # Loading persistent files
    
    import pickle
    import trie

    get_docID = {}
    get_index = {}

    print('Loading data...', end = '')
    data = np.load("mod_data.npy", allow_pickle = True)
    
    print('Done.')

    for i in range(0, len(data)) :
        get_docID[i] = int(data[i][0])
        get_index[int(data[i][0])] = i

    print('Loading tries...', end = '')
    collection = None
    documentRoot = {}
    max_tf = {}
    with open('collection.pickle', 'rb') as handle:
        collection = pickle.load(handle)
    with open('documentRoot.pickle', 'rb') as handle:
        documentRoot = pickle.load(handle)
    with open('max_tf.pickle', 'rb') as handle:
        max_tf = pickle.load(handle)
    print('Done')
    
    user_input = "1"

    while user_input == str(1):

        print('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

        query = input('Enter search query\n')
        tf_query, final_query = preprocess_query(query)
        retrieved_docs = calculate_scores(collection, documentRoot, max_tf, tf_query, final_query)

        print('\n\nTo view a specific document, enter 2. \nTo search for a new query, enter 1.\nTo exit, enter 0.')
        user_input = input('\n')
        
        while user_input != str(0) and user_input != str(1) and user_input !=str(2) :

                user_input=input('\nWrong input. Please re-enter your choice (0,1,2):\n')

        while user_input == str(2):

            while True:
                try:
                    docID_user = input('\nEnter the document ID of the document you wish to view:\n')
                    x = int(docID_user)

                except ValueError:
                    print('\nInvalid input. Please enter a numeric value.')
                    continue
# 
                break

            while int(docID_user) not in retrieved_docs:

                print("\nWrong document ID. Please choose a document ID from among the following:\n")
                for docID in retrieved_docs:
                    print(docID, end=' ')
                print()
                docID_user = input('\nEnter document ID:\n')

            print('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            print()
            print(data[get_index[int(docID_user)]][1])
            print()
            print(data[get_index[int(docID_user)]][4])
            print()
            print('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

            print('\nTo view another document, enter 2.\nTo search for a new query, enter 1.\nTo exit, enter 0.')
            user_input = input('\n')

            while user_input != str(0) and user_input != str(1) and user_input != str(2) :

                user_input=input('\nWrong input. Please re-enter your choice (0,1,2):\n')

