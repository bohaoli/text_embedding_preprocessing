import re
import time
import gensim

from nltk.tokenize import RegexpTokenizer as reg_tokenize
from nltk.corpus import stopwords

class preprocessing_utils():

    def __init__(self, stopwords=set(stopwords.words('english'))):
        self.ave_doc_len_name = "<AVER_DOC_LEN>"
        self.articles = "articles"
        self.paragraphs = "paragraphs"
        self.sentences = "sentences"
        self.stopwords = stopwords

    def modify_word(self, word):
        if word.startswith("-"):
            word = word[1:]
        return word

    def is_valid_word(self, word):
        word = self.modify_word(word.rstrip())
        return word not in self.stopwords and len(word) >= 2

    def my_tokenizer(self, string):
        string = re.sub('\[\d*,*-*\d*\]', "", string)
        tokenizer = reg_tokenize('[a-zA-Z0-9_-]+') #|\$[\d\.]+|\S+   \w+-\d+
        return [self.modify_word(word) for word in tokenizer.tokenize(string.lower()) if self.is_valid_word(word)]

    def compare_model_vocab_and_corpus_vocab(self, w2v_model_name, corpus_voc_file, not_in):
        w2v_model = gensim.models.KeyedVectors.load_word2vec_format(w2v_model_name, binary=(w2v_model_name.endswith('.bin')))
        res = []
        with open(corpus_voc_file, 'r', encoding='utf-8') as f: # the first line of corpus_voc_file is the size of the vocabulary
            i = -1
            start = time.time()
            for line in f:
                i += 1
                if i == 0:
                    continue
                if not_in and line.rstrip() not in w2v_model:
                    res.append(line.rstrip())
                elif not not_in and not line.rstrip().isnumeric() and line.rstrip() in w2v_model:
                    res.append(line.rstrip())
                if i % 50000 == 0:
                    print("It has compared " + str(i) + " words in " + str(time.time() - start) + " seconds")

        # write to result file
        print("The number of the words that are <not_in>" + str(not_in) + " is: " + str(len(res)))
        compare_result_file = "compare-" + w2v_model_name + "-not_in_" + str(not_in) + "-" + corpus_voc_file
        with open(compare_result_file, 'w') as newf:
            newf.write(str(len(res)))
            start = time.time()
            counter = 0
            for word in res:
                counter += 1
                newf.write("\n" + word)
                if counter % 50000 == 0:
                    print("It has writen " + str(counter) + " words in " + str(time.time() - start) + " seconds")

        print()


    def read_vec_file(self, filename):
        with open(filename, 'r', encoding='utf-8') as f:
            i = -1
            for line in f:
                i += 1
                if i == 0:
                    print(line)
                if i % 50000 == 0:
                    print("We have scanned " + str(i) + " lines")
            print("There are " + str(i) + " lines")


if __name__ == "__main__":
    pre = preprocessing_utils()
    # pre.read_vec_file("crawl-300d-2M.txt")
    pre.compare_model_vocab_and_corpus_vocab("crawl-300d-2M.bin", "compare-cord19-300d.bin-not_in_True-index-data_vocab.txt", False)