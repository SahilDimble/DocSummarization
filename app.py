from flask import Flask, render_template, request
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk.data
import nltk
import heapq

app = Flask(__name__)

nltk.download('punkt')
nltk.download('stopwords')

@app.route('/')
def index():
    return render_template('index.html', summary="")

@app.route('/summarize', methods=['POST'])
def summarizer():
    if request.method == 'POST':
        document = request.form['document']
        try:
            # Tokenize the document into sentences and words
            sentences = sent_tokenize(document)
            words = word_tokenize(document)
            
            # Remove stopwords and punctuation
            stop_words = set(stopwords.words('english'))
            word_frequencies = {}
            
            #frequency count of words
            for word in words:
                if word.lower() not in stop_words and word.isalnum():
                    if word.lower() not in word_frequencies:
                        word_frequencies[word.lower()] = 1
                    else:
                        word_frequencies[word.lower()] += 1
            
            # Calculate sentence scores based on word frequencies
            sentence_scores = {}
            for sentence in sentences:
                for word in word_tokenize(sentence.lower()):
                    if word in word_frequencies:
                        if sentence not in sentence_scores:
                            sentence_scores[sentence] = word_frequencies[word]
                        else:
                            sentence_scores[sentence] += word_frequencies[word]
            
            # Get the summary with the top sentences
            summary_sentences = heapq.nlargest(5, sentence_scores, key=sentence_scores.get)
            summary = ' '.join(summary_sentences)
        except Exception as e:
            summary = f"An error occurred: {str(e)}"
        return render_template('index.html', summary=summary)

if __name__ == '__main__':
    app.run(debug=True)
