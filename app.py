#!/usr/bin/env python
# coding: utf-8

# In[11]:

from flask import Flask

import numpy as np

import librosa #feature extraction
import librosa.display

from keras.models import model_from_json

from sklearn.preprocessing import StandardScaler


# In[12]:

app = Flask(__name__)

# In[13]:

def noise(data):
    noise_amp = 0.04*np.random.uniform()*np.amax(data)
    data = data + noise_amp*np.random.normal(size=data.shape[0])
    return data

def stretch(data, rate=0.70):
    return librosa.effects.time_stretch(data, rate)

def shift(data):
    shift_range = int(np.random.uniform(low=-5, high = 5)*1000)
    return np.roll(data, shift_range)

def pitch(data, sampling_rate, pitch_factor=0.8):
    return librosa.effects.pitch_shift(data, sampling_rate, pitch_factor)

def higher_speed(data, speed_factor = 1.25):
    return librosa.effects.time_stretch(data, speed_factor)

def lower_speed(data, speed_factor = 0.75):
    return librosa.effects.time_stretch(data, speed_factor)


# In[14]:


def extract_features(data, sample_rate):
    
    result = np.array([])
    
    mfccs = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=58)
    mfccs_processed = np.mean(mfccs.T,axis=0)
    result = np.array(mfccs_processed)
     
    return result

def get_features(path):
    data, sample_rate = librosa.load(path, duration=3, offset=0.5, res_type='kaiser_fast') 
    
    #without augmentation
    res1 = extract_features(data, sample_rate)
    result = np.array(res1)
    
    #noised
    noise_data = noise(data)
    res2 = extract_features(noise_data, sample_rate)
    result = np.vstack((result, res2)) # stacking vertically
    
    #stretched
    stretch_data = stretch(data)
    res3 = extract_features(stretch_data, sample_rate)
    result = np.vstack((result, res3))
    
    #shifted
    shift_data = shift(data)
    res4 = extract_features(shift_data, sample_rate)
    result = np.vstack((result, res4))
    
    #pitched
    pitch_data = pitch(data, sample_rate)
    res5 = extract_features(pitch_data, sample_rate)
    result = np.vstack((result, res5)) 
    
    #speed up
    higher_speed_data = higher_speed(data)
    res6 = extract_features(higher_speed_data, sample_rate)
    result = np.vstack((result, res6))
    
    #speed down
    lower_speed_data = higher_speed(data)
    res7 = extract_features(lower_speed_data, sample_rate)
    result = np.vstack((result, res7))
    
    return result


# In[15]:
path_ = 'sample_data.wav'

features = get_features(path_)

scaler = StandardScaler()
test_data = scaler.fit_transform(features)
test_data = np.asarray(test_data)
test_data = np.expand_dims(test_data, axis=2)


# In[16]:

# loading json and model architecture 
json_file = open('model_json.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("Emotion_Model.h5")
print("Loaded model from disk")

preds = loaded_model.predict(test_data)


# In[17]:


preds=preds.argmax(axis=1)
high_pred = np.argmax(np.bincount(preds))


# In[19]:

def reverseEncoding(pred):
    emotion = None
    if pred == 0:
        emotion = "angry"
    if pred == 1:
        emotion = "calm"
    if pred == 2:
        emotion = "disgust"
    if pred == 3:
        emotion = "fear"
    if pred == 4:
        emotion = "happy"
    if pred == 5:
        emotion = "neutral"
    if pred == 6:
        emotion = "sad"
    if pred == 7:
        emotion = "surprise"
        
    # Emotion to return
    return emotion

@app.route('/')

def showResult():
    return (reverseEncoding(high_pred))

if __name__ == "__main__":
    app.run(debug=True)
