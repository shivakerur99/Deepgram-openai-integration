// Importing necessary components and libraries from React and other files
import { ThreeCircles } from "react-loader-spinner";
import React, { useState } from 'react';
import "./App.css";
import Api from "./components/Api";

function App() {
// State variables for managing response data, file upload, loading state, user input, and chat log
const [responseData, setResponseData] = useState(null);
const [selectedFile, setSelectedFile] = useState(null);
const [isLoading, setIsLoading] = useState(false);

const [chatLog, setChatLog] = useState([]);

// Function to handle file selection by the user
const handleFileChange = (event) => {
setSelectedFile(event.target.files[0]);
};


 // Function to upload selected file to the server
const uploadFile = async (e) => {
const formData = new FormData();
formData.append('file', selectedFile);
setIsLoading(true);
try {
  const data = await Api.post('/upload/', formData, {
    headers: {
      'Content-Type': 'multipart/form-data'
    }
  });
  setIsLoading(false);
  setResponseData(data.data);
  console.log(data.data)
  console.log('File uploaded successfully');
} catch (error) {
  console.error('Error uploading file:', error);
  setIsLoading(false);
}
};

 // Function to handle form submission
async function handleSubmit(e) {
  e.preventDefault();
   // Update chat log with user input
  // setChatLog([...chatLog, { user: "me", message: input }]);
  // setInput("");
  setIsLoading(true);
  try {
    // Get user input from text area
    const postData = {
      responseData: responseData.content,
    };
     // Send user input to backend and get response
    const response = await Api.post('/doc/', postData, {
      headers: {
        'Content-Type': 'application/json',
      }
    });
    console.log('Response from backend:', response.data);
    // Update chat log with response from backend
    setChatLog([...chatLog,{message: response.data }]);
    setIsLoading(false);
  } catch (error) {
    console.error('Error sending data to backend:', error);
    setIsLoading(false);
  }
}
return (
<div>
  <nav className="nav">
    <div className="nav-logo">
    <img alt="Avatar for ALINDOR" class="rounded-md" height="50" src="https://photos.wellfound.com/startups/i/9988469-734480eb0da9ce8268165d699d84f6fb-medium_jpg.jpg?buster=1707503203" width="50"/>
    <div className="nav-text">ALINDOR</div>
    </div>
    <div className="container">
      <div className="file-logo-container">
        {selectedFile &&
          <svg width="25" height="29" viewBox="0 0 25 29" fill="none" xmlns="http://www.w3.org/2000/svg">
            <rect width="25" height="28.2731" rx="3.386" fill="white" />
            <rect x="0.3386" y="0.3386" width="24.3228" height="27.5959" rx="3.0474" stroke="#0FA958" strokeOpacity="0.44" strokeWidth="0.677201" />
            <path d="M14.1365 6.77197V10.0451C14.1365 10.2621 14.2227 10.4703 14.3761 10.6237C14.5296 10.7772 14.7377 10.8634 14.9548 10.8634H18.2279" stroke="#0FA958" strokeWidth="1.0158" strokeLinecap="round" strokeLinejoin="round" />
            <path d="M16.5914 21.5011H8.40854C7.9745 21.5011 7.55823 21.3287 7.25131 21.0218C6.9444 20.7148 6.77197 20.2986 6.77197 19.8645V8.40854C6.77197 7.9745 6.9444 7.55823 7.25131 7.25131C7.55823 6.9444 7.9745 6.77197 8.40854 6.77197H14.1365L18.228 10.8634V19.8645C18.228 20.2986 18.0555 20.7148 17.7486 21.0218C17.4417 21.3287 17.0254 21.5011 16.5914 21.5011Z" stroke="#0FA958" strokeWidth="1.0158" strokeLinecap="round" strokeLinejoin="round" />
          </svg>}
          <div className="filename">
            {selectedFile && `${selectedFile.name}`}
          </div>
      </div>
      <div className="file-upload-logo-innerfile">
            <label htmlFor="fileInput" className="svg-icon">
              {/* <svg
                width="18"
                height="18"
                viewBox="0 0 18 18"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
                className="svg-icon"
              >
                <g clipPath="url(#clip0_6_746)">
                  <path d="M9 16.875C13.3492 16.875 16.875 13.3492 16.875 9C16.875 4.65076 13.3492 1.125 9 1.125C4.65076 1.125 1.125 4.65076 1.125 9C1.125 13.3492 4.65076 16.875 9 16.875Z" stroke="black" strokeWidth="0.875" strokeLinecap="round" strokeLinejoin="round" />
                  <path d="M5.625 9H12.375" stroke="black" strokeWidth="0.875" strokeLinecap="round" strokeLinejoin="round" />
                  <path d="M9 5.625V12.375" stroke="black" strokeWidth="0.875" strokeLinecap="round" strokeLinejoin="round" />
                </g>
                  <defs>
                    <clipPath id="clip0_6_746">
                      <rect width="18" height="18" fill="white" />
                    </clipPath>
                  </defs>
              </svg> */}
                <input
                  type="file"
                  id="fileInput"
                  onChange={handleFileChange}
                />
            </label>
          </div>
      <div className="Upload-container">
        <div className="file-upload-logo">
          <button className="btn-submit-upload" onClick={uploadFile}>
            Upload txt/mp3
          </button>
        </div>
      </div>
    </div>
  </nav>
  <div className="cont">

  <div className="Analytics">
    <button className="button-send" onClick={handleSubmit}>
          Get Analysis
    </button>
  </div>
  <div className="chat-log">
  {chatLog.map((message, index) => (
    <div key={index}>
      {Array.isArray(message.message) && message.message.map((msg, idx) => {
        if (msg.trim() !== "") {
          const [sentence, description] = msg.split('Description:');
          return (
            <div className="analysis-msg" key={idx}>
              <p>{sentence}</p>
              {description && <p>Analysis: {description}</p>}
            </div>
          );
        } else {
          return null; // Skip rendering if the message is empty
        }
      })}
    </div>
  ))}
</div>
{isLoading && (
          <div className="spinner-container">
            <ThreeCircles color="#0FA958" height={100} width={100} />
          </div>
        )}
  </div>
  </div>
);
}



export default App;

