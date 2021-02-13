from keras.models import load_model

import emoji
import numpy as np




def sentences_to_indices(X, word_to_index, max_len=10):
    """
    Converts an array of sentences (strings) into an array of indices corresponding to words in the sentences.
    The output shape should be such that it can be given to `Embedding()` (described in Figure 4). 
    
    Arguments:
    X -- array of sentences (strings), of shape (m, 1)
    word_to_index -- a dictionary containing the each word mapped to its index
    max_len -- maximum number of words in a sentence. You can assume every sentence in X is no longer than this. 
    
    Returns:
    X_indices -- array of indices corresponding to words in the sentences from X, of shape (m, max_len)
    """
	    
    m = X.shape[0]                                   # number of training examples
    X_indices = np.zeros((m,max_len))
    
    for i in range(m):                               # loop over training examples
        sentence_words =list(X[i].lower().split())
        j = 0
        for w in sentence_words:
            X_indices[i, j] = word_to_index[w]
            j = j+1
    
    return X_indices
def read_glove_vecs(glove_file):
    #glove is stripped to word corpus to reduce the space
    with open(glove_file, 'r',encoding="utf-8") as file:
        
        i = 1
        words_to_index = {}
        for w in list(f.read().strip().split()):
            words_to_index[w] = i
            i = i + 1
    return words_to_index

    
emoji_dictionary = {"0": "\u2764\uFE0F",    # :heart: prints a black instead of red heart depending on the font
                    "1": ":baseball:",
                    "2": ":smile:",
                    "3": ":disappointed:",
                    "4": ":fork_and_knife:"}
def label_to_emoji(label):
    """
    Converts a label (int or string) into the corresponding emoji code (string) ready to be printed
    """
    return emoji.emojize(emoji_dictionary[str(label)], use_aliases=True)
def e_predict(sentence,loaded_model):
    #print("in")
    sentences=np.array(list(sentence.split("."))) 
    #print("sentences = ",sentences)
    word_to_index = read_glove_vecs('data/words_corpus.txt')
    #print("in  1")
    sentence_indices = sentences_to_indices(sentences,word_to_index) 
    #print("in 2")
    #model = load_model("saved_models/model_emojify") 
    #print("model loaded") 
    sent_indi = sentences_to_indices(sentences , word_to_index) 
    preds = loaded_model.predict(sent_indi)  
    preds = np.argmax(preds,axis=1)
    #print("preds=",preds) 
    res="" 
    for i,j in zip(sentences , preds):
        res = res + i+" --> "+label_to_emoji(j)+"\n" 
    return res
