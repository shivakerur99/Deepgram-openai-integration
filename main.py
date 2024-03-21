from flask import Flask, request, render_template, jsonify
from transformers import pipeline
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords
import openai

app = Flask(__name__)
# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
openai.api_key = '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'

# Load the sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis")


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            # Read conversations from the uploaded file
            content = file.read().decode('utf-8')
            conversations = content.split('\n')

        return render_template('results.html')

    return render_template('index.html')

# Sample conversation
conversation = [
    "[Speaker_1]: Hello, Dave. How are you?",
    "[Speaker_2]: Hi, Joseph. I’m good. Yesterday went for a run. What about you?",
    "[Speaker_1]: I’m fine. Today I will read a book. I like reading.",
]

# Function to extract active words from a text with stopwords removed
# def extract_active_words(text):
#     tokens = word_tokenize(text)
#     stop_words = set(stopwords.words('english'))
#     active_words = [word for word in tokens if word.isalnum() and word.lower() not in stop_words]
#     return active_words

# # Analyze sentiment and extract active words for each sentence in the conversation
# sentiments_with_active_words = []
# for sentence in conversation:
#     result = sentiment_pipeline(sentence)
#     sentiment = result[0]['label']
#     active_words = extract_active_words(sentence)
#     sentiments_with_active_words.append((sentiment, active_words))

# # Print the sentiments with active words
# # print("Sentiments and Active Words for each sentence:")
# # for i, (sentence, (sentiment, active_words)) in enumerate(zip(conversation, sentiments_with_active_words)):
# #     print(f"{sentence}: Sentiment: {sentiment}, Active Words: {active_words}")
    
# def generate_description(speaker, sentiment, active_words):
#     prompt = f"{speaker}: Sentiment: {sentiment}\nActive Words: {', '.join(active_words)}\nDescription:"
#     response = openai.Completion.create(
#         engine="gpt-3.5-turbo-instruct",
#         prompt=prompt,
#         temperature=0.7,
#         max_tokens=len(speaker) + 50  # Limit description length to the length of the speaker's text
#     )
#     return response.choices[0].text.strip()

# # Generate descriptions for each sentence
# descriptions = []
# for sentence, (sentiment, active_words) in zip(conversation, sentiments_with_active_words):
#     # Extract speaker name
#     speaker = sentence.split(":")[0]
#     description = generate_description(speaker, sentiment, active_words)
#     descriptions.append(description)

# # Print the descriptions
# print("Generated Descriptions for each sentence:")
# for i, (sentence, description) in enumerate(zip(conversation, descriptions)):
#     print(f"Sentence {i+1}: {sentence}")
#     print(f"Description: {description}\n")

if __name__ == '__main__':
    app.run(debug=True)




# from flask import Flask, request, render_template, jsonify
# from transformers import pipeline
# from nltk.tokenize import word_tokenize
# import nltk
# from nltk.corpus import stopwords
# import openai

# app = Flask(__name__)

# # Download NLTK resources
# nltk.download('punkt')
# nltk.download('stopwords')

# # Set OpenAI API key
# openai.api_key = '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'

# # Load the sentiment analysis pipeline
# sentiment_pipeline = pipeline("sentiment-analysis")

# # Function to extract active words from a text with stopwords removed
# def extract_active_words(text):
#     tokens = word_tokenize(text)
#     stop_words = set(stopwords.words('english'))
#     active_words = [word for word in tokens if word.isalnum() and word.lower() not in stop_words]
#     return active_words

# # Function to generate description using OpenAI's GPT-3 model
# def generate_description(speaker, sentiment, active_words):
#     prompt = f"{speaker}: Sentiment: {sentiment}\nActive Words: {', '.join(active_words)}\nDescription:"
#     response = openai.Completion.create(
#         engine="gpt-3.5-turbo-instruct",
#         prompt=prompt,
#         temperature=0.7,
#         max_tokens=len(speaker) + 50  # Limit description length to the length of the speaker's text
#     )
#     return response.choices[0].text.strip()

# @app.route('/', methods=['GET', 'POST'])
# def upload_file():
#     if request.method == 'POST':
#         file = request.files['file']
#         if file:
#             # Read conversations from the uploaded file
#             content = file.read().decode('utf-8')
#             conversations = content.split('\n')
            
#             # Analyze sentiment, extract active words, and generate descriptions
#             descriptions = []
#             for sentence in conversations:
#                 result = sentiment_pipeline(sentence)
#                 sentiment = result[0]['label']
#                 active_words = extract_active_words(sentence)
#                 speaker = sentence.split(":")[0]
#                 description = generate_description(speaker, sentiment, active_words)
#                 descriptions.append(description)
            
#             return render_template('results.html', descriptions=descriptions)

#     return render_template('index.html')

# if __name__ == '__main__':
#     app.run(debug=True)


# from flask import Flask, request, render_template
# from transformers import pipeline
# from nltk.tokenize import word_tokenize
# import nltk
# from nltk.corpus import stopwords
# import openai

# app = Flask(__name__)

# # Download NLTK resources
# nltk.download('punkt')
# nltk.download('stopwords')

# # Set OpenAI API key
# openai.api_key = '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'

# # Load the sentiment analysis pipeline
# sentiment_pipeline = pipeline("sentiment-analysis")

# # Function to extract active words from a text with stopwords removed
# def extract_active_words(text):
#     tokens = word_tokenize(text)
#     stop_words = set(stopwords.words('english'))
#     active_words = [word for word in tokens if word.isalnum() and word.lower() not in stop_words]
#     return active_words

# # Function to generate description using OpenAI's GPT-3 model
# def generate_description(conversations):
#     descriptions = []
#     for sentence in conversations:
#         result = sentiment_pipeline(sentence)
#         sentiment = result[0]['label']
#         active_words = extract_active_words(sentence)
#         speaker = sentence.split(":")[0]
#         prompt = f"{speaker}: Sentiment: {sentiment}\nActive Words: {', '.join(active_words)}\nDescription:"
#         response = openai.Completion.create(
#             engine="gpt-3.5-turbo-instruct",
#             prompt=prompt,
#             temperature=0.7,
#             max_tokens=len(speaker) + 50  # Limit description length to the length of the speaker's text
#         )
#         description = response.choices[0].text.strip()
#         descriptions.append(description)
#     return descriptions

# @app.route('/', methods=['GET', 'POST'])
# def upload_file():
#     if request.method == 'POST':
#         file = request.files['file']
#         if file:
#             # Read conversations from the uploaded file
#             content = file.read().decode('utf-8')
#             conversations = content.split('\n')
#             print(conversations)
            
#             # Generate descriptions for conversations
#             descriptions = generate_description(conversations)
            
#             return render_template('results.html', descriptions=descriptions)

#     return render_template('index.html')

# if __name__ == '__main__':
#     app.run(debug=True)

