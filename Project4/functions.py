def get_lyrics(df):
    '''
    Returns a dataframe containing the lyrics for the artists songs of the given dataset
    '''
    from bs4 import BeautifulSoup
    import requests
    import time

    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.198 Safari/537.36"}
    lyrics_list = []
    count = 0

    for pos in range(len(df)):
        success = False
        respose = None
        while success == False:
            
            response = requests.get(df.Link[pos], headers)
            if response.status_code == 429:
                print("error", response.status_code)
                time.sleep(2)
            else:
                success = True

        soup = BeautifulSoup(markup = response.text)
        result = soup.find(name = "pre", attrs= {"class" : "lyric-body"})#.text.splitlines()
        if result is None:
            print("cant find pre:", count)
            lyrics_list.append("no text")
            count += 1
            pass
        else:
            success = True
            lyrics_temp = result.text.splitlines()            
            lyrics_list.append(lyrics_temp)
            print(count, "success!")
            count += 1
                

    return lyrics_list


def list_to_chunks(artistname, song_sum):
    '''
    A function that turns all lyrics into chucks of 500 words
    
    Input: 
    - artistname = as written in the dataframe (song_sum) 
    - song_sum = dataframe of artist names and all their songs in form of a list of lists (each song as individual list)

    one cell for list of lists of songs of one artist
    1) next step is to make list of lists into one list, 
    2) then concatenate text into one string and 
    3) split the string by single words

    check length of shortest song. -> min length
    chose 5 lines and concatenate into one str -> per song multiple 5 line chunks of song
    for each artist on their own, make chucks of lyrics in one list of lists
    '''
    n = 200
    import re
    from itertools import chain
    songs = song_sum.loc[artistname]["Lyrics"]
    flat = list(chain(*songs))
    string = ",".join(flat).lower()
    one_string = string.replace(",", " ").replace("'", "").replace("  ", " ").replace(r"[0-9]", "").replace(r"[^a-zA-Z]", "").replace("(", "").replace(")", "")
    words_list = one_string.split()
    chunks = [" ".join(words_list[i:i + n]) for i in range(0, len(words_list), n)]

    return chunks


def most_occ_words(artistname):
    '''
    Returns a list of artist and their associated most occuring words of their lyrics
    '''
    from collections import Counter
    import re
    from itertools import chain
    songs = song_sum.loc[artistname]["Lyrics"]
    flat = list(chain(*songs))
    string = ",".join(flat).lower()
    one_string = string.replace(",", " ").replace("'", "").replace("  ", " ").replace(r"[0-9]", "").replace(r"[^a-zA-Z]", "").replace("(", "").replace(")", "")
    words_list = one_string.split()
    c = Counter(words_list)
    most_occur = c.most_common(5)
    most_occ = pd.DataFrame(most_occur)

    return most_occ


def train_model(text, labels):
    '''for naive bayes'''
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import make_pipeline

    tf_vec = TfidfVectorizer()
    nb = MultinomialNB(alpha = 1) # for count data like the one we have with the words in texts
    model = make_pipeline(tf_vec, nb) 
    model.fit(text, labels)
    
    return model
 

def predict(model, new_text):
    '''
    for naive bayes
    
    input: 
    - model: the trained model
    - new_text: a string of lyrics
    '''

    new_text = [new_text]
    prediction = model.predict(new_text)
    probabilities = model.predict_proba(new_text)
    
    return prediction[0], probabilities



def print_evaluations(y_true, y_pred, model):
    '''
    A funciton that prints the confusion matrix of the actual and predicted probabilities of the respective model

    input: 
    - y_true: the true probability
    - y_pred: the predicted probability
    - model: the modelname as a string
    '''

    import numpy as np
    import sklearn.metrics as metrics
    import matplotlib.pyplot as plt
    import seaborn as sns

    print(f'How does model {model} score:')
    print(f'Accuracy:' + str(np.round(metrics.accuracy_score(y_true, y_pred), 3)))
    print(f'Precision:' + str(np.round(metrics.precision_score(y_true, y_pred, average="weighted"), 3)))
    print(f'Recall:' + str(np.round(metrics.recall_score(y_true, y_pred, average="weighted"), 3)))
    print(f'F1:' + str(np.round(metrics.f1_score(y_true, y_pred, average="weighted"), 3)))
    
    #print confusion matrix
    fig = plt.figure(figsize=(6, 6))
    cm = metrics.confusion_matrix(y_true, y_pred)
    print(cm)
    
    #plot the heatmap
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax); 
    
    # labels, title and ticks
    ax.set_xlabel('Predicted labels');
    ax.set_ylabel('True labels'); 
    ax.set_title('Confusion Matrix')