'''
This program calculate the idf score of each word in the corpus
'''
import json
import os
import time
import math
import shelve

from preprocessing_utils import preprocessing_utils
from collections import defaultdict
import nltk.data

class idf_score():

    def __init__(self, shelve_file_name, data_set, pre_utils, by_sent_para_doc, to_store, to_create_vocab):
        self.data_set = data_set
        self.by_sent_para_doc = by_sent_para_doc # The value is "by_sent", "by_para" or "by_doc", which indicates what we consider as a document
        self.idf = defaultdict(lambda: 0)
        self.preprocessing_utils = pre_utils
        self.corpus_size = 0
        self.total_doc_length = 0
        self.average_document_length = 0
        self.shelve_file_name = shelve_file_name
        self.to_store = to_store # Will we store the idf-score to a shelve file
        self.to_create_vocab = to_create_vocab # Will we create a vocabulary based on this data-set

    # Given a list of dictionaries, with each one contains "text", "cite_spans", "ref_spans", "section", return a list of sentences.
    # Each sentence is a string.
    def extract_text_from_paragraphs(self, list_of_paragraphs):
        res = []
        if self.by_sent_para_doc == self.preprocessing_utils.articles:
            res = self.preprocessing_utils.my_tokenizer(" ".join(list_of_paragraphs))
        elif self.by_sent_para_doc == self.preprocessing_utils.paragraphs:
            for para in list_of_paragraphs:
                res.append(self.preprocessing_utils.my_tokenizer(para))
        elif self.by_sent_para_doc == self.preprocessing_utils.sentences:
            for para in list_of_paragraphs:
                tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
                lst_of_sentences = tokenizer.tokenize(para) #tokenizer.tokenize(text_witout_spans)
                for sentence in lst_of_sentences:
                    res.append(self.preprocessing_utils.my_tokenizer(sentence))
        return res

    '''
    Given a file, extract corpus from the file according to the text type we want.
    '''
    def extract_corpus(self, filename):
        res = []
        document = [] # This would only be usful if self.by_sent_para_doc = "by_doc"
        with open(filename, encoding='utf-8') as f:
            data = json.load(f)
            # append title
            if "title" in data:
                if self.by_sent_para_doc == "by_doc":
                    document += self.preprocessing_utils.my_tokenizer(data['title'])
                else:
                    res.append(self.preprocessing_utils.my_tokenizer(data['title']))

            # append abstract
            if "textAbstract" in data:                
                if self.by_sent_para_doc == "by_doc":
                    document += self.extract_text_from_paragraphs(data['textAbstract'])
                else:
                    res += self.extract_text_from_paragraphs(data['textAbstract'])

            #append body text
            if "bodyText" in data:
                if self.by_sent_para_doc == "by_doc":
                    document += self.extract_text_from_paragraphs(data['bodyText'])
                else:
                    res += self.extract_text_from_paragraphs(data['bodyText'])
        if self.by_sent_para_doc == "by_doc":
            res.append(document)
        return res

    '''
    Given a list of corpus, count the number of words in this corpus and store the information 
    to the dictionary
    '''
    def add_corpus(self, corpus_buffer):
        print("length of corpus buffer is: " + str(len(corpus_buffer)))
        self.corpus_size += len(corpus_buffer)
        for document in corpus_buffer:
            word_set = set()
            self.total_doc_length += len(document)
            for word in document:
                if word not in word_set:
                    self.idf[word] += 1
                    word_set.add(word)

    '''
    calculate the df of the corpus
    '''

    def initialize_df(self):
        corpus_buffer = []
        counter = 0
        print("Start to get corpus")
        start = time.time()
        for root, dirs, files in os.walk(self.data_set):
            for name in files:
                counter += 1
                corpus_buffer += self.extract_corpus(os.path.join(root, name))
                if counter % 1000 == 0:
                    self.add_corpus(corpus_buffer)
                    print("It takes us " + str(time.time() - start) + " seconds to add " + str(counter) + " files, the size of our idf is " + str(len(self.idf)))
                    corpus_buffer = []
        self.add_corpus(corpus_buffer)
        corpus_buffer = []

    '''
    Calculate the idf score of each word in this corpus
    '''
    def calculate_idf(self):
        sentence = "In the search for a vaccine against SARS"
        for word in self.preprocessing_utils.my_tokenizer(sentence):
            print("The freq of " + word + " is: " + str(self.idf[word]))

        for word in self.idf:
            freq = self.idf[word]
            self.idf[word] = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)

    # Store a dictionary to a python shelve
    def store_with_shelve(self):
        counter = 0
        start = time.time()
        with shelve.open(self.shelve_file_name) as s:
            for key in self.idf:
                counter += 1
                s[key] = self.idf[key]
                if counter % 5000 == 0:
                    print("It takes us " + str(time.time() - start) + " seconds to store " + str(counter) + " words to shelve")
            s[self.preprocessing_utils.ave_doc_len_name] = self.average_document_length

    '''
    Create the vocabulary of the corpus
    '''
    def create_vocab(self, vocab_file_name):
        with open(vocab_file_name, 'w') as f:
            f.write(str(len(self.idf)))
            for word in self.idf:
                f.write("\n" + word)

    '''
    Create idf score of the corpus. May or may not create vocabulary or store the idf score to shelve files.
    '''
    def preprocessing(self):
        if self.to_store:
            with shelve.open(self.shelve_file_name) as s:
                if len(s) > 0:
                    return
        self.initialize_df()
        print("The size of our idf is: " + str(len(self.idf)))
        self.calculate_idf()
        self.average_document_length = self.total_doc_length / self.corpus_size
        self.idf[self.preprocessing_utils.ave_doc_len_name] = self.average_document_length
        print("The average_document_length is: " + str(self.average_document_length))

        if self.to_create_vocab:
            self.create_vocab(self.data_set + "_vocab.txt")

        if self.to_store:
            print("Start to store with shelve")
            start = time.time()
            self.store_with_shelve()
            print("It takes " + str(time.time() - start) + " seconds to store to shelve")

if __name__ == "__main__":
    dataset = "medium"
    pre_utils = preprocessing_utils()
    text_type = pre_utils.paragraphs
    # constructor of idf_score
    # def __init__(self, shelve_file_name, data_set, pre_utils, by_sent_para_doc, to_store, to_create_vocab)
    wei = idf_score("idf_score_" + text_type, dataset, pre_utils, text_type, True, False)
    start = time.time()
    wei.preprocessing()
    end = time.time()
    print("It takes " + str(end - start) + " to preprocessing")