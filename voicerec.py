# Import the necessary libraries
import os
import glob
import numpy as np
import soundfile
import librosa
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import joblib

# Emotions in the RAVDESS dataset
emotions = {
'01':'neutral',
'02':'calm',
'03':'happy',
'04':'sad',
'05':'angry',
'06':'fearful',
'07':'disgust',
'08':'surprised'
}

# Emotions we want to observe
observed_emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

# Function to extract features from a sound file
def extract_feature(file_name, mfcc, chroma, mel):
    # Read the sound file
    with soundfile.SoundFile(file_name) as sound_file:
        # Get the sound file's data and sample rate
        X = sound_file.read(dtype="float32")
        sample_rate=sound_file.samplerate

        # If Chroma is one of the desired features, we calculate the STFT (Short-Time Fourier Transform) of the sound file
        # "chroma" refers to the 12 different pitch classes
        if chroma:
            stft=np.abs(librosa.stft(X))

        # Initialize an empty numpy array for storing the features
        result=np.array([])

        # Extract MFCC feature and append it to the result array
        if mfcc:
            mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result=np.hstack((result, mfccs))

        # Extract Chroma feature and append it to the result array
        if chroma:
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result=np.hstack((result, chroma))

        # Extract Mel Spectrogram feature and append it to the result array
        if mel:
            mel=np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T,axis=0)
            result=np.hstack((result, mel))

    return result

# Function to load the data and extract features for each sound file
def load_data(test_size=0.2):
    # Initialize empty lists for storing the features and labels
    x, y = [], []

    # Iterate over all the sound files in the RAVDESS dataset
    for file in glob.glob("ravdess/Actor_*/*.wav"):
        # Get the filename
        file_name = os.path.basename(file)

        # Extract the emotion label from the filename
        emotion = emotions[file_name.split("-")[2]]

        # If the emotion label is not in the list of observed emotions, we ignore this file
        if emotion not in observed_emotions:
            continue

        # Extract features from the sound file
        feature = extract_feature(file, mfcc=True, chroma=True, mel=True)

        # Append the features and label to their respective lists
        x.append(feature)
        y.append(emotion)

    # Split the data into training and testing sets
    return train_test_split(np.array(x), y, test_size=test_size, random_state=9)

# Function to train the voice emotion recognition model
def train_voice():
    # Load the data and split it into training and testing sets
    x_train,x_test,y_train,y_test = load_data(test_size=0.25)

    # Instantiate a Label Encoder object
    lb = LabelEncoder()

    # Fit and transform the labels in the training set to encoded versions, then convert these to categorical format
    y_train = to_categorical(lb.fit_transform(y_train))
    # Transform the labels in the test set to encoded versions, then convert these to categorical format
    y_test = to_categorical(lb.transform(y_test))

    # Save the trained Label Encoder to a file for later use in predictions
    joblib.dump(lb, 'label_encoder.joblib')
    
    model_file_path = 'model.h5'
    print ('x_train.shape--------------\n', x_train.shape)
    # Check if a trained model already exists
    if os.path.exists(model_file_path):
        print("Loading existing model")
        # Load the pre-existing model
        model = load_model(model_file_path)
    else:
        # If a model does not already exist, we need to create and train one
        # Construct the architecture of the model
        model = Sequential([
            Dense(256, activation='relu', input_shape=(x_train.shape[1],)), # Input layer with 256 nodes
            Dense(128, activation='relu'), # Hidden layer with 128 nodes
            Dense(64, activation='relu'), # Hidden layer with 64 nodes
            Dense(len(observed_emotions), activation='softmax'), # Output layer with a number of nodes equal to the number of emotions we are recognizing, softmax is for multi-class classification
        ])

        # Compile the model with appropriate loss function, optimizer, and metrics
        # categorical_crossentropy is for multi-class classification problems
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model using the training data
    history = model.fit(x_train, y_train, epochs=50, validation_data=(x_test, y_test))
    
    # List all data in history
    print('training result-----------\n', history.history.keys())
    
    # Save the trained model to a file for later use
    model.save(model_file_path)

    # Load the model that we have just trained
    model = load_model(model_file_path)

    # Evaluate the model's performance using the test data
    loss, accuracy = model.evaluate(x_test, y_test)

    # Print the accuracy of the model
    print("Accuracy: {:.2f}%".format(accuracy*100))

# Function to evaluate the voice emotion recognition model on a single audio file
def evaluate_voice():
    model_file_path = 'model.h5'

    # Check if the specified model file exists
    if not os.path.isfile(model_file_path):
        print("File 'model.h5' not found.")
        return    
    
    # Load the previously trained model
    model = load_model(model_file_path)
    
    label_file_path = 'label_encoder.joblib'

    # Check if the specified label encoder file exists
    if not os.path.isfile(label_file_path):
        print("File 'label_encoder.joblib' not found.")
        return    
    
    # Load the previously fitted Label Encoder
    lb = joblib.load('label_encoder.joblib')
    
    file = 'test.wav'

    # Check if the specified audio file exists
    if not os.path.isfile(file):
        print("File 'test.wav' not found.")
        return
    
    # Extract features from the audio file
    feature = extract_feature(file, mfcc=True, chroma=True, mel=True)
    
    # Reshape the features to be [1, number_of_features] because we are predicting the class for one sample
    feature = np.array([feature])
    
    # Use the model to predict the class for the features
    result = model.predict(feature)
    
    # The output of the model is a probability distribution over classes, so we take the class with the highest probability
    predicted_class = np.argmax(result)
    
    # Reverse transform the encoded label to get the original emotion
    emotion = lb.inverse_transform([predicted_class])
    
    print("The predicted emotion is ", emotion[0])
    
#train_voice()
evaluate_voice()
