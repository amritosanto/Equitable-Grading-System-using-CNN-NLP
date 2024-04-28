import h5py
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.models import load_model


model = load_model('ml_model.h5')


with h5py.File('ml_dataset.h5', 'r') as hf:
    if 'vocabulary' in hf:
        vocabulary = hf['vocabulary'][:]

        vocabulary = [word.decode('utf-8') for word in vocabulary]
    else:
        vocabulary = None

vectorizer = CountVectorizer(vocabulary=vocabulary)


if vocabulary is not None:
    vectorizer.fit(vocabulary)


def predict_marks(text):

    text_encoded = vectorizer.transform([text])


    if text.strip() == '' or np.sum(text_encoded) == 0:
        return 0
    

    marks_predicted = model.predict(text_encoded)
    
    if marks_predicted > 10:
        scaled_marks = 10
    else:
        scaled_marks = marks_predicted
    
    return scaled_marks[0][0] if isinstance(scaled_marks, np.ndarray) else scaled_marks



text = "Machine learning is a subset of artificial intelligence (AI) that focuses on enabling machines to learn from data without being explicitly programmed. It involves creating algorithms that can learn patterns and make predictions or decisions based on input data. These algorithms improve their performance over time as they are exposed to more data. Machine learning techniques are used in various fields, including healthcare, finance, marketing, and more, to solve complex problems and make data-driven decisions. There are several types of machine learning, including supervised learning, unsupervised learning, and reinforcement learning, each suited for different tasks and data types."
predicted_marks = predict_marks(text)
print("Predicted Marks:", predicted_marks)
