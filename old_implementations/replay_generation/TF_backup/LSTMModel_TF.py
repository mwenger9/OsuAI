from keras.models import Model
from keras.layers import Dense, LSTM, Input, GaussianNoise

class LSTMModel:
    def __init__(self, num_input_features, num_output_features):
        self.num_input_features = num_input_features
        self.num_output_features = num_output_features
        self.model = self._build_model()
    
    def _build_model(self):
        map_input = Input(shape=(None, self.num_input_features), name='map_info')

        lstm = LSTM(64, return_sequences=True)(map_input)
        pos = Dense(64, activation='linear')(lstm)
        pos = GaussianNoise(0.2)(pos)
        pos = Dense(16, activation='linear')(pos)
        pos = Dense(self.num_output_features, activation='linear', name='position')(pos)

        model = Model(inputs=map_input, outputs=pos)
        model.compile(optimizer='adam', loss='mae')

        return model
    
    def summary(self):
        self.model.summary()

    def load_weights(self, filepath):
        try:
            self.model.load_weights(filepath)
            print("Loaded weights from disk")
        except Exception as e:
            print(f"Failed to load weights: {e}")
