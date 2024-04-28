import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


with h5py.File('ml_dataset.h5', 'r') as hf:
    encoded_answers = hf['encoded_answers'][:]
    labels = hf['labels'][:]


X_train, X_test, y_train, y_test = train_test_split(encoded_answers, labels, test_size=0.2, random_state=42)


model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')
])


model.compile(optimizer='SGD', loss='mean_squared_error')


model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)


loss = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')


model.save('ml_model.h5')
