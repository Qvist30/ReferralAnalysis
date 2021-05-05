from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import twitter_samples, stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk import FreqDist, classify, NaiveBayesClassifier
from collections import Counter
from nrclex import NRCLex
from gensim.corpora import Dictionary
from gensim.models import LdaModel
import gensim.corpora as corpora
import numpy


import re, string, random

def remove_noise(tweet_tokens, stop_words = ()):

    cleaned_tokens = []

    for token, tag in pos_tag(tweet_tokens):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
        token = re.sub("(@[A-Za-z0-9_]+)","", token)

        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)

        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens

def get_all_words(cleaned_tokens_list):
    for tokens in cleaned_tokens_list:
        for token in tokens:
            yield token

def get_tweets_for_model(cleaned_tokens_list):
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tweet_tokens)
def prep_phrase(phrase):
    '''
    Removes stopwords, punctuation from text, and converts into a list of word tokens
    
    Args:
    phrase = text string
    
    Outputs:
    list of word tokens
    '''
    
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(phrase)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    return filtered_sentence

def topic_scorer(phrase, topic, sim_thresh = 0.6, return_hits = False):
    '''
    For each word in a sentence, retrieves the synonym set. For each synonym we measure the wup_similarity
    to the topic at hand. If similarity > sim_threshold, the topic is said to have been mentioned.
    The wup_similarity threshold can be configured: where a higher threshold for increases the strictness of the word-to-topic similarity condition.
    If return_hits is set to True, the words in the phrase that were mapped to each topic will be returned.
    
    Args:
    filtered_sentence = tokenized sentence, preferrably stripped of stopwords
    topic = synset of the topic in question.
    sim_thresh = wup_similarity threshold for word and topic to be deemed similar enough (default 0.6)
    return_hits = return the words that matched to each topic (default = False)
    
    Outputs:
    Integer count of the number of mentions of the topic in the filtered_sentence
    '''
    
    phrase = prep_phrase(phrase)
    word_scores = []
    
    for w in range(len(phrase)):
        syns = wordnet.synsets(phrase[w])
        syns_sim = [topic.wup_similarity(syns[synonym]) for synonym in range(len(syns))]
        syns_sim = [sim if sim is not None else 0 for sim in syns_sim]
        try:
            syns_sim = np.max([1 if sim > sim_thresh else 0 for sim in syns_sim])
        except ValueError:
            syns_sim = 0
        word_scores.append(syns_sim)
    hits = [phrase[w] for w in range(len(phrase)) if word_scores[w] == 1]
        
    if return_hits:    
        return (np.sum(word_scores), hits)
    else:
        return np.sum(word_scores)        
        
def multi_topic_scorer(phrase, topic_dictionary, sim_thresh=0.6, return_hits=False):
    '''
    Takes a passage of text and maps words in that text to topics that have been defined in a topic dictionary.
    The wup_similarity threshold can be configured: where a higher threshold for increases the strictness of the word-to-topic similarity condition.
    If return_hits is set to True, the words in the phrase that were mapped to each topic will be returned.
    
    Args:
    phrase = passage of text
    topic_dictionary = dictionary where key:value is reader-friendly topic name:assigned synonym in wordnet
    sim_thresh = wup_similarity threshold for word and topic to be deemed similar enough (default 0.6)
    return_hits = return the words that matched to each topic (default = False)
    
    Outputs:
    sim_scores = dictionary where key:value is the reader-friendly topic name:number of synonyms present in the text
    '''

    sim_scores = {}
    
    for topic in list(topic_dictionary.keys()):
        topic_synset = wordnet.synset(topic_dictionary['{}'.format(topic)])
        sim_scores['{}'.format(topic)] = topic_scorer(phrase, topic_synset, sim_thresh, return_hits)
    return sim_scores        

if __name__ == "__main__":

#     positive_tweets = twitter_samples.strings('positive_tweets.json')
#     negative_tweets = twitter_samples.strings('negative_tweets.json')
#     text = twitter_samples.strings('tweets.20150430-223406.json')
#     tweet_tokens = twitter_samples.tokenized('positive_tweets.json')[0]

#     stop_words = stopwords.words('english')

#     positive_tweet_tokens = twitter_samples.tokenized('positive_tweets.json')
#     negative_tweet_tokens = twitter_samples.tokenized('negative_tweets.json')

#     positive_cleaned_tokens_list = []
#     negative_cleaned_tokens_list = []

#     for tokens in positive_tweet_tokens:
#         positive_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

#     for tokens in negative_tweet_tokens:
#         negative_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

#     all_pos_words = get_all_words(positive_cleaned_tokens_list)

#     freq_dist_pos = FreqDist(all_pos_words)
#     print(freq_dist_pos.most_common(10))

#     positive_tokens_for_model = get_tweets_for_model(positive_cleaned_tokens_list)
#     negative_tokens_for_model = get_tweets_for_model(negative_cleaned_tokens_list)

#     positive_dataset = [(tweet_dict, "Positive")
#                          for tweet_dict in positive_tokens_for_model]

#     negative_dataset = [(tweet_dict, "Negative")
#                          for tweet_dict in negative_tokens_for_model]

#     dataset = positive_dataset + negative_dataset

#     random.shuffle(dataset)

#     train_data = dataset[:7000]
#     test_data = dataset[7000:]
    

#     classifier = NaiveBayesClassifier.train(train_data)

#     print("Accuracy is:", classify.accuracy(classifier, test_data))

#     print(classifier.show_most_informative_features(10))

    with open ("/Users/thomaskennedy/downloads/interview1.txt", "r") as myfile:
        file_tokens = myfile.readlines()
        
#     print(custom_tweet)

    topic_dictionary = {'Real Estate': 'real_property.n.01',
                    'Accounting': 'accounting.n.02',
                    'Financial Services' : 'fiscal.a.01'}
    
    
    #custom_tweet = "YES, i think yes think A bat cat loves think yes car car yes"
    wordsInFile = numpy.array([]);
    for line in file_tokens:
        line = line.replace("Mike Mazzei: ","")
        custom_tokens = remove_noise(word_tokenize(line))
        
        if(line[0].isdigit() == False and line.startswith("Jacob") == False and len(line.strip()) != 0):
            
            # This is positive/negative classifier
            #print(classifier.classify(dict([token, True] for token in custom_tokens)), " : ", line)
            wordsInFile = numpy.append(wordsInFile, custom_tokens)
            
            #Cheer up EMOLec
            text_object = NRCLex(line)
            print(line)
            print(text_object.raw_emotion_scores)
            print(multi_topic_scorer(line, topic_dictionary, sim_thresh=0.7, return_hits=True))


#     print(custom_tokens)
    
    
#     with open ("/Users/thomaskennedy/downloads/interview1.txt", "r") as myfile:
#         custom = myfile.readlines()
#     print(wordsInFile)

    Counter = Counter(wordsInFile)
    # most_common() produces k frequently encountered
    # input values and their respective counts.
    most_occur = Counter.most_common(50)
    
    for occur in most_occur:
        print(occur);
    
    #Note that this data is unclean, need to use something other than file_tokens, probably
    text_object = NRCLex(' '.join(file_tokens))

    print(text_object.raw_emotion_scores)

    print(multi_topic_scorer(' '.join(file_tokens), topic_dictionary, sim_thresh=0.7, return_hits=True))
  
    
#     print()
    
#     print(classifier.show_most_informative_features(10))
