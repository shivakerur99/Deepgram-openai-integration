from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, File, UploadFile
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String
from databases import Database


import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import openai
import time
nltk.download('stopwords')
openai.api_key = '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'

app = FastAPI()

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
        prompt="do not mention sentiment and active words in description only on basis of sentiment and Active words give required description description should be like this example [Speaker_1]: Hello, Dave. How are you? [Speaker_2]: Hi, Joseph. I’m good. Yesterday went for a run. What about you? [Speaker_1]: I’m fine. Today I will read a book. I like reading. · If it’s an audio file for Hero, Master, Grandmaster modes, it can be any recording with at least two speakers, such as a short fragment from a podcast or interview. So, in input of Web interface Hiring manager attaches text or audio with any human conversation. In output Hiring manager should get sentiment or psychological insights derived from the conversation, some insights about speakers. Please don’t provide summary of conversation, key words, etc. Output should be related to sentimental analysis. For example: Line 1 in web interface: ‘[Speaker_2] likes a sport. It seems he cares about his health’.Line 2 in web interface: ‘[Speaker_1] pretends to be smart’. "+prompt,
        temperature=0.7,
        max_tokens=len(speaker) + 50  # Adjusted to a fixed value for simplicity
    )
    return response.choices[0].text.strip()

# Endpoint for uploading text files
@app.post("/upload/")
async def upload_text_file(file: UploadFile = File(...)):
    # Check if the uploaded file is a text file
    if not file.filename.lower().endswith('.txt'):
        raise HTTPException(status_code=400, detail="Only text files (TXT) are allowed.")
    
    # Read the content of the file asynchronously
    contentinitial = await file.read()
    contentlast = contentinitial.decode('utf-8')
    filtered_content = '\n'.join(line for line in contentlast.splitlines() if line.strip())
    content=filtered_content
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

    # It's good practice to close the file after processing
    await file.close()

    return doc

class DataInput(BaseModel):
    responseData: str

@app.post("/doc/")
async def process_data(data: DataInput):
    # Access responseData and userInput
    content = data.responseData
    conversation = parse_conversation(content)
    sentiments_with_active_words = []
    for sentence in conversation:
    # Using OpenAI's sentiment analysis API
        result = openai.Completion.create(
            engine="gpt-3.5-turbo-instruct", 
            prompt=sentence + " sentiment:",
            temperature=0,
            max_tokens=1,
            n=1,
            stop=None,
        )
        sentiment = result['choices'][0]['text'].strip()
        time.sleep(15)
        # Extract active words
        active_words = extract_active_words(sentence)
        
        sentiments_with_active_words.append((sentiment, active_words))
    
    descriptions = []
    for sentence, (sentiment, active_words) in zip(conversation, sentiments_with_active_words):
        speaker = sentence.split(":")[0]
        time.sleep(12)  # Reduced sleep time for demonstration; adjust as per rate limits
        description = generate_description(speaker, sentiment, active_words)
        descriptions.append(description)


    print("Generated Descriptions for each sentence:")
    l=[]
    for i, (sentence, description) in enumerate(zip(conversation, descriptions)):
        l.append(f"Sentence {i+1}: {sentence}\n")
        l.append(f"Description: {description}\n")

    
    return l