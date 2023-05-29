from LSTMModel_TF import LSTMModel
from DataGenerator_TF import DataGenerator
from sklearn.model_selection import train_test_split
import pandas as pd 


def train_model(model, data_generator, val_data_generator, epochs, model_save_path):
    # Initialize best_loss to infinity
    best_loss = float('inf')

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        # Training loop
        for i in range(len(data_generator)):
            X_train, y_train = data_generator[i]
            model.model.fit(X_train, y_train)

        # Validation loop
        val_loss = 0
        for i in range(len(val_data_generator)):
            X_val, y_val = val_data_generator[i]
            loss = model.model.evaluate(X_val, y_val)
            val_loss += loss

        # Compute average validation loss
        avg_val_loss = val_loss / len(val_data_generator)
        print(f"Validation loss: {avg_val_loss}")

        # If the validation loss is the best we've seen so far, save the model weights
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            model.model.save_weights(model_save_path)
            print("Saved model to disk")


if __name__ == "__main__":

    DATASET_PATH = "D:\\osu!rdr_dataset"
    beatmap_folder = f"{DATASET_PATH}\\beatmaps\\"
    replay_folder = f"{DATASET_PATH}\\replays\\osr\\"
    df = pd.read_csv(f'{DATASET_PATH}\\final_cleaned_index.csv')


    train_df, validation_df = train_test_split(df, test_size=0.2)

    batch_size = 128

    training_data_generator = DataGenerator(train_df, beatmap_folder, replay_folder, batch_size=batch_size)
    validation_data_generator = DataGenerator(validation_df, beatmap_folder, replay_folder, batch_size=batch_size)

    num_input_features = 162

    num_output_features = 5
    my_model = LSTMModel(num_input_features, num_output_features)
    my_model.summary()
    my_model.load_weights("best_model.h5")

    train_model(my_model, training_data_generator, validation_data_generator, 10, 'best_model.h5')
