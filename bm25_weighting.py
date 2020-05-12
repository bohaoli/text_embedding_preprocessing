'''
This program is to create bm25 scores of a corpus
'''
import time
import shelve

from idf_score import idf_score
from preprocessing_utils import preprocessing_utils
from collections import defaultdict

class bm25_weighting():

    def __init__(self, idf_file, k_value, b_value, utils, idf_object, use_avg=False):
        self.preprocessing_utils = utils
        self.idf_score = {}
        if idf_object != None:
            self.idf_score = idf_object.idf
        else:
            self.idf_score = shelve.open(idf_file)# It takes time. We'd better run the preprocessing method of an idf_score object before hand
        self.average_doc_length = self.idf_score[utils.ave_doc_len_name]
        self.k = k_value
        self.b = b_value
        self.use_avg = use_avg

    '''
    Givent the word count and length of the word, return the best match score of the bm25 score in this document
    '''
    def best_match(self, cwd, len_d):
        return (self.k + 1) * cwd / (cwd + self.k * (1 - self.b + self.b * len_d / self.average_doc_length))

    '''
    Given a document, return a dictionary whose keys are words in the document and values are the bm25 score of the words
    '''
    def get_bm25_weight(self, document):
        word_list = self.preprocessing_utils.my_tokenizer(document)
        word_freq = defaultdict(lambda: 0)
        total_num_words = 0
        for word in word_list:
            if word in self.idf_score:
                total_num_words += 1
                word_freq[word] += 1
        for word in word_freq:
            word_freq[word] /= total_num_words
        if self.use_avg:
            for word in word_freq:
                word_freq[word] = 1
        else:
            for word in word_freq:
                word_freq[word] = self.idf_score[word] * self.best_match(word_freq[word], total_num_words)
        return word_freq

# This is the main funtion
if __name__ == "__main__":
    start = time.time()
    idf_score_tmp = idf_score("idf_score", "index-data", preprocessing_utils(), False, False, False)
    idf_score_tmp.preprocessing()
    bm25wei = bm25_weighting("idf_score", 1.2, 0.75, preprocessing_utils(), idf_score_tmp)
    end = time.time()
    print("It takes " + str(end - start) + " seconds to create a bm25wei")
    res = []
    start = time.time()
    res.append(bm25wei.get_bm25_weight("Infection by many enveloped viruses requires fusion of the viral and cellular membranes."))
    res.append(bm25wei.get_bm25_weight("Although rhinoviruses have been thought to be transmitted via large droplets or contact."))
    res.append(bm25wei.get_bm25_weight("An impediment to future studies of airborne transmission of respiratory infections is the lack of established methods for the detection of airborne respiratory microorganisms appropriate for epidemiological studies."))
    res.append(bm25wei.get_bm25_weight("To determine if airborne microorganisms can be inactivated, and therefore, prevent infection transmission, studies during the 1940's–1970's used ultraviolet germicidal irradation (UVGI) to inactivated airborne microorganisms under experimental conditions."))
    res.append(bm25wei.get_bm25_weight("Uranine was used as a fluorescent tracer, which allowed us to determine a percent yield for each sampling run."))
    res.append(bm25wei.get_bm25_weight("The amount of uranine nebulized during each sampling run was determined by measuring the amount of fluid consumed and by measuring the fluorescence of nebulizing fluid before and after each sampling run."))
    res.append(bm25wei.get_bm25_weight("We computed the expected number of TCID50 on each filter sample from the known TCID50 and fluid volume added to the nebulizer, the amount of nebulizing fluid consumed per minute, and the sampling time."))
    res.append(bm25wei.get_bm25_weight("We were able to obtain from the dilution gradient filter spiking experiments a positive band from a 1:500,000 dilution of virus, which translates into a detected amount of approximately 0.77 TCID50 or 0.39 PFU's per filter."))
    res.append(bm25wei.get_bm25_weight("In order to determine if the uranine in the viral suspension was inhibiting the extraction and detection assay, two otherwise identical runs were conducted – one with and one without uranine."))
    res.append(bm25wei.get_bm25_weight("Infection by many enveloped viruses requires fusion of the viral and cellular membranes."))
    res.append(bm25wei.get_bm25_weight("Infection by many enveloped viruses requires fusion of the viral and cellular membranes."))
    end = time.time()
    print("It takes " + str(end - start) + " seconds to calculate bm25 of a sentence")
    for item in res:
        print(item)

