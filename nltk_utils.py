import numpy as np
import nltk
# nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
from nltk import pos_tag  # Nhập hàm pos_tag
stemmer = PorterStemmer()


def tokenize(sentence):
    """
    split sentence into array of words/tokens
    a token can be a word or punctuation character, or number
    """
    return nltk.word_tokenize(sentence)


def stem(word):
    """
    stemming = find the root form of the word
    examples:
    words = ["organize", "organizes", "organizing"]
    words = [stem(w) for w in words]
    -> ["organ", "organ", "organ"]
    """
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, words):
    """
    return bag of words array:
    1 for each known word that exists in the sentence, 0 otherwise
    example:
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bog   = [  0 ,    1 ,    0 ,   1 ,    0 ,    0 ,      0]
    """
    # stem each word
    sentence_words = [stem(word) for word in tokenized_sentence]
    # initialize bag with 0 for each word
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1

    return bag


def pos_weighted_bag_of_words(tokenized_sentence, words):
    """
    trả về mảng bag of words có trọng số dựa trên POS tagging:
    1 cho mỗi từ đã biết tồn tại trong câu, được trọng số theo thẻ POS của nó
    """
    # gốc hóa từng từ
    sentence_words = [stem(word) for word in tokenized_sentence]
    # khởi tạo bag với 0 cho mỗi từ
    bag = np.zeros(len(words), dtype=np.float32)
    
    # Lấy thẻ POS cho câu đã token hóa
    pos_tags = pos_tag(tokenized_sentence)
    
    for idx, w in enumerate(words):
        if w in sentence_words:
            # Trọng số có thể được điều chỉnh dựa trên thẻ POS
            pos = dict(pos_tags)[w]  # Lấy thẻ POS cho từ
            weight = 1.0  # Trọng số mặc định
            if pos.startswith('NN'):  # Ví dụ: Danh từ
                weight = 1.5
            elif pos.startswith('VB'):  # Ví dụ: Động từ
                weight = 1.2
            bag[idx] = weight

    return bag  # Trả về bag of words có trọng số
