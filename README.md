# ChatBot_NLU
Natural Language Understanding using in this chatbot 
The main use to implement this chabtot is to understsand the interaction of user and computer by analysis of words


NLU chatbots are widely used in various applications such as customer service, virtual assistants, information retrieval, and more, offering users a more natural and intuitive way to interact with computer systems.


This code implements a simple chatbot using TensorFlow/Keras for natural language processing and Tkinter for the graphical user interface (GUI). Here's a summary of its functionality:


1.Importing Libraries: The necessary libraries are imported, including TensorFlow/Keras for the model, NLTK for text processing, and Tkinter for GUI.

2.Loading Data: The code loads pre-trained model weights, preprocessed data (words and classes), and intents from JSON files.

3.Text Preprocessing: It defines functions to clean up sentences by tokenizing and lemmatizing them, and converting them into a bag of words representation.

4.Intent Prediction: The predict_class function predicts the intent class of a given sentence using the trained model. It calculates the probability of each class and selects those with probabilities above a certain threshold.

5.Generating Responses: The get_response function selects a response based on the predicted intent. It randomly selects a response from the list of responses corresponding to the predicted intent.

6.GUI Setup: Tkinter is used to create a simple GUI with a text area for displaying chat history, an entry field for user input, and a button to send messages.

7.Message Sending: The send_message function is called when a message is sent (either by pressing the 'Send' button or hitting 'Return' in the entry field). It gets the user's message, 

8.predicts the intent, generates a response, and updates the chat history accordingly.

Program Execution: The main window is created, GUI components are packed, and the event loop (root.mainloop()) is started to run the GUI application.
