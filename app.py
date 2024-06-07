from flask import Flask, render_template, request
import pickle
import numpy as np
import spacy
import nltk
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Load the NLP model and VADER sentiment analyzer
nlp = spacy.load("en_core_web_sm")
nltk.download('vader_lexicon')
sid = SentimentIntensityAnalyzer()

# Load your pre-trained model
#with open('nlp_model3.pkl', 'rb') as model_file:
#    model = pickle.load(model_file)

app = Flask(__name__)

def preProcess(paragraph):
    doc = nlp(paragraph)
    processed_sentences = []
    for sentence in doc.sents:
        tokens = [token.lemma_.lower() for token in sentence if
                  not token.is_stop and not token.is_punct and not token.is_space]
        processed_sentence = " ".join(tokens)
        processed_sentences.append(processed_sentence)
    return processed_sentences

def findNumberOfSentences(text):
    doc = nlp(text)
    num_sentences = len(list(doc.sents))
    return num_sentences

def checkNumerical(sentence):
    doc = nlp(sentence)
    for token in doc:
        if token.like_num:
            try:
                a = float(token.text)
                return 1
            except ValueError:
                pass
    return 0

def containNumerical_data(text):
    arr = []
    for sentence in text:
        has_numerical = checkNumerical(sentence)
        arr.append(has_numerical)
    return arr

def check_similarity(sentence1, sentence2):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform([sentence1, sentence2])
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return similarity[0][0]

def comparing_sentences(sentences1, sentences2):
    similarity_matrix = np.zeros((len(sentences2), len(sentences1)))
    for i, sentence1 in enumerate(sentences1):
        for j, sentence2 in enumerate(sentences2):
            similarity_score = check_similarity(sentence1, sentence2)
            similarity_matrix[j][i] = similarity_score

    for i in range(len(sentences2)):
        max_similarity_index = np.argmax(similarity_matrix[i])
        max_similarity_score = similarity_matrix[i][max_similarity_index]
        similarity_matrix[i] = 0
        similarity_matrix[i][max_similarity_index] = max_similarity_score
    return similarity_matrix

def extract_keywords(sentence):
    doc = nlp(sentence)
    keywords = []
    for token in doc:
        if token.pos_ in ['NOUN', 'PROPN', 'ADJ']:
            keywords.append(token.text)
    return keywords

def determine_context(sentence1, sentence2):
    keywords1 = extract_keywords(sentence1)
    keywords2 = extract_keywords(sentence2)

    common_keywords = set(keywords1).intersection(keywords2)

    if common_keywords:
        keyword_counts = Counter(common_keywords)
        return max(keyword_counts, key=keyword_counts.get)
    else:
        return 0

def get_sentiment_score(sentence):
    sentiment_score = sid.polarity_scores(sentence)
    return sentiment_score['compound']

def compare_sentences(sentence1, sentence2):
    sentiment_score1 = get_sentiment_score(sentence1)
    sentiment_score2 = get_sentiment_score(sentence2)

    if sentiment_score1 > sentiment_score2:
        return 1
    elif sentiment_score1 < sentiment_score2:
        return -1
    else:
        return 0

def extract_numerical_values(sentence):
    doc = nlp(sentence)
    numerical_values = []
    for token in doc:
        if token.pos_ == 'NUM':
            try:
                numerical_values.append(float(token.text))
            except ValueError:
                pass
    return max(numerical_values)

def compare_numerical_values(sentence1, sentence2):
    values1 = extract_numerical_values(sentence1)
    values2 = extract_numerical_values(sentence2)

    if not values1 or not values2:
        return "No numerical values found in one or both sentences"

    max_value1 = values1
    max_value2 = values2

    if max_value1 == max_value2:
        return "0"
    elif max_value1 > max_value2:
        return "1"
    else:
        return "-1"

def main(p1, p2):
    processedP1 = preProcess(p1)
    processedP2 = preProcess(p2)
    matrix2d = comparing_sentences(processedP1, processedP2)
    isNum1 = containNumerical_data(processedP1)
    isNum2 = containNumerical_data(processedP2)
    rows = np.count_nonzero(matrix2d) + 2
    columns = 4
    finalMatrix = np.empty((rows, columns), dtype=np.dtype('U50'))
    finalMatrix[0][1] = "Text1"
    finalMatrix[0][2] = "Text2"
    finalMatrix[0][3] = "Output"
    finalMatrix[rows - 1][0] = "Total"

    for i in range(0, rows - 2):
        for j in range(0, findNumberOfSentences(p1)):
            if matrix2d[i][j] != 0:
                common_keyword = determine_context(processedP2[i], processedP1[j])
                finalMatrix[i + 1][0] = common_keyword
                if isNum1[j] != 0 and isNum2[i] != 0:
                    if get_sentiment_score(processedP1[j]) > 0 and get_sentiment_score(processedP2[i]) > 0:
                        finalMatrix[i + 1][1] = extract_numerical_values(processedP1[j])
                        finalMatrix[i + 1][2] = extract_numerical_values(processedP2[i])
                        finalMatrix[i + 1][3] = compare_numerical_values(processedP1[j], processedP2[i])
                    elif get_sentiment_score(processedP1[j]) > 0 and get_sentiment_score(processedP2[i]) < 0:
                        finalMatrix[i + 1][1] = extract_numerical_values(processedP1[j])
                        finalMatrix[i + 1][2] = extract_numerical_values(processedP2[i])
                        finalMatrix[i + 1][3] = "1"
                    elif get_sentiment_score(processedP1[j]) < 0 and get_sentiment_score(processedP2[i]) > 0:
                        finalMatrix[i + 1][1] = extract_numerical_values(processedP1[j])
                        finalMatrix[i + 1][2] = extract_numerical_values(processedP2[i])
                        finalMatrix[i + 1][3] = "-1"
                    else:
                        finalMatrix[i + 1][1] = extract_numerical_values(processedP1[j])
                        finalMatrix[i + 1][2] = extract_numerical_values(processedP2[i])
                        finalMatrix[i + 1][3] = -1 * int(compare_numerical_values(processedP1[j], processedP2[i]))
                else:
                    finalMatrix[i + 1][1] = get_sentiment_score(processedP1[j])
                    finalMatrix[i + 1][2] = get_sentiment_score(processedP2[i])
                    finalMatrix[i + 1][3] = compare_sentences(processedP1[j], processedP2[i])
    sum = 0
    for i in range(1, rows - 1):
        sum = sum + int(finalMatrix[i][columns - 1])
    finalMatrix[rows - 1][columns - 1] = sum

    result = finalMatrix.tolist()

    # Printing Final decision
    decision = ""
    if finalMatrix[rows - 1][columns - 1] > "0":
        decision = "Entity in Text 1 is better than Entity in Text 2"
    elif finalMatrix[rows - 1][columns - 1] < "0":
        decision = "Entity in Text 2 is better than Entity in Text 1"
    else:
        decision = "Entity in Text 1 is equal to Entity in Text 2"

    return result, decision

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/compare', methods=['POST'])
def compare():
    paragraph1 = request.form['paragraph1']
    paragraph2 = request.form['paragraph2']
    result, decision = main(paragraph1, paragraph2)
    return render_template('index.html', result=result, decision=decision)

if __name__ == '__main__':
    app.run(debug=True)
