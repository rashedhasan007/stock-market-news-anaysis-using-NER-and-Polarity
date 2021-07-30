
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def sentiment_scores(sentence):


    sid_obj = SentimentIntensityAnalyzer()

    sentiment_dict = sid_obj.polarity_scores(sentence)

    print("Overall sentiment dictionary is : ", sentiment_dict)
    print("sentence was rated as ", sentiment_dict['neg' ] *100, "% Negative")
    print("sentence was rated as ", sentiment_dict['neu' ] *100, "% Neutral")
    print("sentence was rated as ", sentiment_dict['pos' ] *100, "% Positive")

    print("Sentence Overall Rated As", end = " ")

    # decide sentiment as positive, negative and neutral
    if sentiment_dict['compound'] >= 0.05 :
        print("Positive")

    elif sentiment_dict['compound'] <= - 0.05 :
        print("Negative")


    else :
        print("Neutral")
    return sid_obj.polarity_scores(sentence)

