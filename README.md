# Alindor-app-

# Frontend

first create your virtual environment in your machine using command [vitualenv "name of your env"]
then activate your environment using command ["nameofyourenvironment"\Scripts\activate] for deactivation of your environment run command ["nameofyourenvironment"\Scripts\deactivate]

first for frontend go to alindor-app folder
cmd [cd alindor-app]
run command [npm install]
next step
run command [npm start]

go to Api.js and mention your backend server running on which port and server it is running on

# Backend

first create your virtual environment in your machine using command [vitualenv "name of your env"]
then activate your environment using command ["nameofyourenvironment"\Scripts\activate] for deactivation of your environment run command ["nameofyourenvironment"\Scripts\deactivate]

first for Backend go to FastApi folder

cmd [ cd FastApi]

and install all the necessary packages using command

pip install -r requirements.txt 

then run command [uvicorn main:app --reload]

# Explorer mode
precaution = Kindly don't use long files or long audio files of more than 1 minute as i am using an open AI free credits plan, if we hit the daily limit of 200 requests per day it will throw an error  also so use less lengthy audio i have provided sample audio file in GitHub repo for this
# steps to run the app for [deployed]


first load backend server "https://alindor-ev3t.onrender.com" (wait until it loads)

and load frontend server "https://alindor-1.onrender.com/" (it load fastly)

click on + button to upload file and then (important) click on upload txt 

click on get analysis button wait till it process (if we use 3 lines txt file then 2 mins to process as I am using the free version so I added time.sleep function in code to process it, if we use subscribed version we can increase request per minute)


video of working app in local machine and vc also =https://drive.google.com/file/d/1-lu2LiJWXrkRaMw-eJqGbxTQsXTvTmgX/view?usp=sharing

Average solution
as I want to share my insights on this (important)
-> As I am using Openai free trial I can request 3 API calls per minute so if the .txt file has 3 lines then 3 API calls for sentimental analysis and 3 API calls for getting psychological insights for sentences based on sentimental analysis, require time to process is 2 minute (above link is the live app demo for this implementation)

Worst solution
as the assignment says try Openai 
-> So I tried to process all sentences in data.txt in one API call for sentimental analysis but the results were inefficient and not good, I tried to process in a single API call to get psychological insights for sentences present in .txt files but the results are not efficient(mixing of sentences)

Best solution
-> This is the best approach I have til now for this app to process fast 
for sentimental analysis in Python TextBlob is a Python (2 and 3) library for processing textual data. which is used for sentimental analysis, As I checked the performance of TextBlob with Openai's sentimental analysis results are pretty similar to it so we can use this instead of Openai, just only for sentimental analysis. To get psychological insights for sentences we can still use Openai and process time will be reduced by half.

# Hero and Master mode

(I am combining these two modes because they are almost similar only condition was to deploy on the cloud for master mode)
The GitHub repo will be the same as the above one so 

I used text blob for sentimental analysis and for extracting active words present in sentence I used nltk stopwords and providing prompt based on that 

kindly first load the backend URL then load front end URL

Deployed app frontend link = https://alindor-front2.onrender.com/

Deployed app Backend link = https://alindor-hm.onrender.com

(as I am providing the backend link so you can check Fastapi swagger documentation and test it out here also)

video of working app in local machine and vc also = https://drive.google.com/file/d/15fHWwbo9sC7TUaQ3zDOYXWeBOIr-T7jh/view?usp=sharing

challenges I faced during this app
 
I need to improve process time than Explorer mode so basically in Explorer mode I used Openai's chat completion for both sentimental analysis and for prompt also. due to the limit of 3 requests per minute as per Openai's free plan, my first challenge was how to come up with a solution to process only 3 requests in one minute so I used time.sleep after every request it sleeps for 20 sec. I reduced process time by half using the Python package Textblob for sentimental which has similar efficiency to Openai's sentimental analysis.
deepgram API is not that accurate in terms of speaker diarization






