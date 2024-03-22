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


# steps to run the app for [deployed]

first load backend server "https://alindor-ev3t.onrender.com" (wait until it loads)

and load frontend server "https://alindor-1.onrender.com/" (it load fastly)

click on + button to upload file and then (important) click on upload txt 

click on get analysis button wait till it process (if we use 3 lines txt file then 2 mins to process as i am using free version so i added time sleep in code to process it, if we used subscribed version we can increase request per minute)

