# Local-PDF_Chatbot-using-LLM (Using FAISS)

This application is tested on PDF documents which can be ingested and necessary answers can be seen when a local Large Language Model (LLM) is loaded onto main memory. This is my first project on Prompt Engineering. 
This includes an ingestion based file to convert documents to vector embedded chunks, a model mapping file which loads the LLMs using CTransformers library running under 'CPU' mode and the main file - chat.py which includes a streamlit application where the application is run locally after calling in terminal from VSCode.

The application's running device is "CPU", so the llm infernece would be more and response would be generated according your RAM (min. 16GB).

# Installation 

Create a new anaconda environment of the name you want to give. Then,refer to the file named pipinstalls.txt to follow the pip installs the modules you can also see the llm models link, copy paste the link on search bar and download files.

You can also use the file "requirements_ver.txt" after referring to this vid- https://youtu.be/Tu_HcflONZM?si=ri8zHQ6mrCP31TE-

# Documents Folder

In this folder add necessary PDFs you would like to work with, I have added paper 2 and paper 3 as reference you can check it as test case after cloning repo into local machine.

# Model Folder

For the application we need the local LLMs file under Models folder, download the models from huggingface website using this link https://huggingface.co/models.
The model list is in llm models.txt file

# Run Application

To run the application, use the command line:

"streamlit run chat.py"

# Application page

This is streamlit application in which on the left pane includes the model selection and controlled parameter selection defined under CTransformers library. The main pane includes the chat between the bot and we the users.
To run the application, set the model name and the parameters range, the responses are generated based on the model selection and parameters. For ex: Temperature is preferred to be in the range of 0 to 1.

<p align="center">
  <img src="https://github.com/mr-nobody15/Local-PDF_Chatbot/assets/70313481/89dd2b88-6e0f-4712-ad93-7ee5b0cd9df8" alt="Image Description" height="400">
</p>

<p align="center">
  <img src="https://github.com/mr-nobody15/Local-PDF_Chatbot/assets/70313481/7468ac15-f5b6-4dae-9e87-fd453a879f6d" alt="Image Description" height="370">
</p>

# Outputs 

Below output response is given by "mistral-7b-instruct":

<p align="center">
  <img src="https://github.com/mr-nobody15/Local-PDF_Chatbot/assets/70313481/4f037a57-1875-48ea-a595-6de1871cbd4a" alt="image" width="800" height="340">
</p>

Below output response is given by "Vicuna-13 5-bit":

<p align="center">
  <img src="https://github.com/mr-nobody15/Local-PDF_Chatbot/assets/70313481/c0d0e5ca-1e86-4192-a616-70f83a7e64a3" alt="image" width="800" height="340">
</p>


