from typing import List
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, File, UploadFile
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String
from databases import Database
from textblob import TextBlob


import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import openai
import time
nltk.download('punkt')
nltk.download('stopwords')

openai.api_key = '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'

app = FastAPI()

import os

import requests
import json

# Replace 'YOUR_DEEPGRAM_API_KEY' with your actual API key
API_KEY = '4fab359b7ff314b1fc7a49bd34e389384b00d5ad'

# Define the API endpoint
url = 'https://api.deepgram.com/v1/listen'

# Set request headers
headers = {
    'Authorization': f'Token {API_KEY}',
    'Content-Type': 'audio/wav'  # Ensure the correct content type for your audio file
}

# Set request parameters
params = {
    'diarize': 'true',
    'punctuate': 'true',
    'utterances': 'true',
}

# Set up CORS (Cross-Origin Resource Sharing) for allowing requests from all origins
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Define SQLAlchemy engine and metadata
DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(DATABASE_URL)
metadata = MetaData()

# Define the document table schema
documents = Table(
    "documents",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("filename", String),
    Column("upload_date", String),
    Column("content", String),
)

# Create the document table in the database
metadata.create_all(engine)

# Define Pydantic model for the document
class Document(BaseModel):
    filename: str
    upload_date: str
    content: str

# Initialize database connection pool
database = Database(DATABASE_URL)



def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    if sentiment > 0:
        return "positive"
    elif sentiment < 0:
        return "negative"
    else:
        return "neutral"

def analyze_conversation_sentiment(conversation):
    sentiment_analysis = {}
    for line in conversation:
        speaker, dialogue = line.strip().split(':')
        sentiment = analyze_sentiment(dialogue)
        sentiment_analysis[line] = sentiment
    return sentiment_analysis


def parse_conversation(content):
    return content.strip().split('\n')


def extract_active_words(text):
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    active_words = [word for word in tokens if word.isalnum() and word.lower() not in stop_words]
    return active_words


def generate_description(speaker, sentiment, active_words):
    prompt = f"{speaker}: Sentiment: {sentiment}\nActive Words: {', '.join(active_words)}\nDescription:"
    response = openai.Completion.create(
        engine="gpt-3.5-turbo-instruct",
        prompt=prompt+"do not mention sentiment and active words in description, In output based on sentiment get psychological insights derived from the conversation, some insights about speakers. Please don’t provide summary of conversation, key words, etc. Output should be related to sentimental analysis.",
        temperature=0.7,
        max_tokens=len(speaker) + 50  # Adjusted to a fixed value for simplicity
    )
    return response.choices[0].text.strip()

# Endpoint for uploading text or mp3 or wav files

    
@app.post("/upload/")
async def upload_text_file(file: UploadFile = File(...)):
    # Check if the uploaded file is a text file
    if not file.filename.lower().endswith(('.txt', '.mp3', '.wav')):
        raise HTTPException(status_code=400, detail="Only text files (TXT) or mp3 or wav are allowed.")

    # Define the file path to save the uploaded file in the current directory
    file_path = os.path.join(os.getcwd(), file.filename)

    if file.filename.lower().endswith('.txt'):
        # Read the content of the file asynchronously
        contentinitial = await file.read()
        contentlast = contentinitial.decode('utf-8')
        filtered_content = '\n'.join(line for line in contentlast.splitlines() if line.strip())
        content = filtered_content
        print(content)

    elif file.filename.lower().endswith((".mp3", ".wav")):
        # Save the uploaded audio file in the current directory
        with open(file_path, 'wb') as audio_file:
            audio_file.write(await file.read())

        # Initialize an empty string to store the transcript
        full_transcript = ""

        # Make the API request
        response = requests.post(url, headers=headers, params=params, data=open(file_path, 'rb'))

        # Check if request was successful
        if response.ok:
            # Parse JSON response
            data = response.json()
            # Extract and concatenate transcripts with speaker information into a single string
            for utterance in data['results']['utterances']:
                speaker = utterance['speaker']
                transcript = utterance['transcript']
                full_transcript += f'[Speaker: {speaker}] {transcript}\n'
        else:
            print(f"Error: {response.status_code} - {response.reason}")
        content = full_transcript
        print(content)

        

    # Create document object
    doc = Document(filename=file.filename, upload_date=str(datetime.now()), content=content)

    # Insert the document data into the database
    async with database.transaction():
        query = documents.insert().values(
            filename=doc.filename,
            upload_date=doc.upload_date,
            content=doc.content
        )
        last_record_id = await database.execute(query)

    return doc

class DataInput(BaseModel):
    responseData: str

@app.post("/doc/")
async def process_data(data: DataInput):
    # Access responseData and userInput
    content = data.responseData
    conversation = parse_conversation(content)
    sentiments_with_active_words = []


    #IMPORTANT KINDLY READ IT:
    #IMPORTANT KINDLY READ IT:
    #IMPORTANT KINDLY READ IT:
    #IMPORTANT KINDLY READ IT:  
    #********************OpenAI sentiment analysis part which takes to many api request calls to process big files *****************************#
    # for sentence in conversation:
    # # Using OpenAI's sentiment analysis API
    #     result = openai.Completion.create(
    #         engine="gpt-3.5-turbo-instruct", 
    #         prompt=sentence + " sentiment:",
    #         temperature=0,
    #         max_tokens=1,
    #         n=1,
    #         stop=None,
    #     )
    #     sentiment = result['choices'][0]['text'].strip()
    #     time.sleep(20)
    #     # Extract active words
    #     active_words = extract_active_words(sentence)
    
    

    #********************sentiment analysis Using Textblob which is good and effecient and efficiency match with OpenAI's sentimental analysis********************#
    sentiment_analysis = analyze_conversation_sentiment(conversation)
    for line, sentiment in sentiment_analysis.items():
        active_words = extract_active_words(line)       
        sentiments_with_active_words.append((sentiment, active_words))
    
    # print(sentiments_with_active_words)
    descriptions = []
    for sentence, (sentiment, active_words) in zip(conversation, sentiments_with_active_words):
        speaker = sentence.split(":")[0]
        time.sleep(20)  # Reduced sleep time for demonstration; adjust as per rate limits
        description = generate_description(speaker, sentiment, active_words)
        descriptions.append(description)


    print("Generated Descriptions for each sentence:")
    l=[]
    for i, (sentence, description) in enumerate(zip(conversation, descriptions)):
        l.append(f"Sentence {i+1}: {sentence}\n")
        l.append(f"Description: {description}\n")

    
    return l
    
