import numpy as np
import scipy.linalg as scipy_linalg
import re
import json
import os
import time
import gensim

from nltk.tokenize import RegexpTokenizer as reg_tokenize
from bm25_weighting import bm25_weighting
from preprocessing_utils import preprocessing_utils
from idf_score import idf_score
import nltk.data

class sentence_embedding():

    def __init__(self, w2v_file, dimension, pre_utils, weighting, text_type, weight_of_title=3):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(w2v_file, binary=True)
        self.dimension = dimension
        self.num_of_sentences = 0
        self.preprocessing_utils = pre_utils
        self.bm25_weighting = weighting
        self.text_type = text_type
        self.weight_of_title = weight_of_title

    def get_sentence_to_vector(self, sentence):
        res = np.zeros((self.dimension))
        counter = 0
        curr_weighting = self.bm25_weighting.get_bm25_weight(sentence)
        for word in curr_weighting:
            if word in self.model:
                counter += 1
                res += self.model[word] * curr_weighting[word]
        return [counter, res]

    def sentence_to_vector(self, sentence, sentence_List):
        res = self.get_sentence_to_vector(sentence)
        if res[0] > 0:
            sentence_List.append((sentence, res[1].tolist()))

    def get_similrity(self, sentence1, sentence2):
        v1 = self.get_sentence_to_vector(sentence1)
        v2 = self.get_sentence_to_vector(sentence2)
        res = 0
        if v1[0] > 0 and v2[0] > 0:
            word1_vector = v1[1]
            word2_vector = v2[1]
            res = np.dot(word1_vector, word2_vector) / (scipy_linalg.norm(word1_vector) * scipy_linalg.norm(word2_vector))
        print("The similarity of " + sentence1 + " and " + sentence2 + " is " + str(res))

    # Given a list of dictionaries, with each one contains "text", "cite_spans", "ref_spans", "section", return a list of sentences.
    # Each sentence is a string.
    def extract_text_from_paragraphs(self, list_of_paragraphs):
        res = []
        for para in list_of_paragraphs:
            tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
            lst_of_sentences = tokenizer.tokenize(para) #tokenizer.tokenize(text_witout_spans)
            res += lst_of_sentences
        return res

    # split one document into several documents according to our self.text_type. 
    # Now we only support the following text types: 
    # preprocessing_utils.articles, 
    # preprocessing_utils.paragraphs,
    # preprocessing_utils.sentences
    def split_document(self, data):
        res = []
        # split by document, just merge different parts of a document and return it
        if self.text_type == self.preprocessing_utils.articles:
            document = ""
            # append title
            if "title" in data:
                for i in range(self.weight_of_title):
                    document += data['title']
                    document += " "
            # append abstract
            if "textAbstract" in data:
                for para in data['textAbstract']:
                    document += para
                    document += " "
            #append body text
            if "bodyText" in data:
                for para in data['bodyText']:
                    document += para
                    document += " "
            res.append(document)
        # split by paragraph, append each paragraph to the result list
        elif self.text_type == self.preprocessing_utils.paragraphs:
            # append title
            if "title" in data:
                res.append(data['title'])
            # append abstract
            if "textAbstract" in data:
                for para in data['textAbstract']:
                    res.append(para)
            #append body text
            if "bodyText" in data:
                for para in data['bodyText']:
                    res.append(para)
        # split by sentence, extract sentences from paragraphs and append each sentence to the result list
        elif self.text_type == self.preprocessing_utils.sentences:
            # append title
            if "title" in data:
                res.append(data['title'])
            # append abstract
            if "textAbstract" in data:
                res += self.extract_text_from_paragraphs(data['textAbstract'])
            #append body text
            if "bodyText" in data:
                res += self.extract_text_from_paragraphs(data['bodyText'])
        else:
            print("Invalid text type")

        return res

    def embed(self, data):
        res = []
        documents = self.split_document(data)
        for document in documents:
            self.sentence_to_vector(document, res)
        self.num_of_sentences += len(res)
        return res

    def write_to_json(self, input_file, output_file, doc_id):
        date = "-"
        sentence_vetors = []
        with open(input_file, encoding='utf-8') as f:
            data = json.load(f)
            if "publishTime" in data:
                date = data["publishTime"]
            sentence_vetors = self.embed(data)
        dictionary_lst = []
        for pair in sentence_vetors:
            curr_dict = {}
            curr_dict["sentence"] = pair[0]
            curr_dict["vector"] = pair[1]
            curr_dict["publishTime"] = date
            dictionary_lst.append(curr_dict)
        res = {}
        res[doc_id.split(".")[0]] = dictionary_lst
        json_object = json.dumps(res)
        with open(output_file, "w") as output:
            output.write(json_object)

    def sentence_embed_files(self, data_set, output_dir):
        file_counter = 0
        start = time.time()
        for root, dirs, files in os.walk(data_set):
            for name in files:
                file_counter += 1
                self.write_to_json(os.path.join(root, name), output_dir + name + ".em.josn", name)
                if file_counter % 1000 == 0:
                    end = time.time()
                    print("It takes " + str(end - start) + " seconds to embed " + str(file_counter) + " files")


if __name__ == "__main__":
    dataset = "medium" # folder of preprocessed article files
    pre_utils = preprocessing_utils()
    text_type = pre_utils.paragraphs # "sentences", "paragraphs", "articles"
    # constructor of idf_score 
    # def __init__(self, shelve_file_name, data_set, pre_utils, by_sent_para_doc, to_store, to_create_vocab)
    idf_score_tmp = idf_score("idf_score_" + text_type, dataset, pre_utils, text_type, False, False)
    idf_score_tmp.preprocessing()
    bm25wei = bm25_weighting("idf_score_" + text_type, 1.2, 0.75, pre_utils, idf_score_tmp)
    start = time.time()
    se = sentence_embedding("cord19-300d.bin", 300, pre_utils, bm25wei, text_type) 
    path = text_type + "_embedding_300d-data/"
    try:
        os.mkdir(path)
    except OSError:
        print ("Creation of the directory %s failed" % path)
    else:
        print ("Successfully created the directory %s " % path)
    se.sentence_embed_files(dataset, path)
    print("There are " + str(se.num_of_sentences) + text_type + "!")
    end = time.time()
    print("It takes " + str(end - start) + " seconds")