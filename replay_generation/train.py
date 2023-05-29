import torch
from torch import nn
from LSTMModel import LSTMModel
from DatasetLoader import DataGenerator
import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    NUM_EPOCHS = 1
    
    DATASET_PATH = "D:\\osu!rdr_dataset"
    beatmap_folder = f"{DATASET_PATH}\\beatmaps\\"
    replay_folder = f"{DATASET_PATH}\\replays\\osr\\"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    df = pd.read_csv(f'{DATASET_PATH}\\final_cleaned_index.csv')

    df = df.sample(n=1000)

    train_df, validation_df = train_test_split(df, test_size=0.2)

    batch_size = 8
    print("Generating the training and validation sets")
    training_data_generator = DataGenerator(train_df, beatmap_folder, replay_folder, batch_size=batch_size)
    validation_data_generator = DataGenerator(validation_df, beatmap_folder, replay_folder, batch_size=batch_size)

    num_input_features = 161
    num_output_features = 5
    print("Creating the model")
    model = LSTMModel(input_size=num_input_features, hidden_size=50, num_layers=1, output_size=num_output_features)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    print(f"Using device {device}")
    model = model.to(device)

    print("Start of the training loop")
    # Training loop
    for epoch in range(NUM_EPOCHS):
        print(f"Starting epoch {epoch}")
        for i, (inputs, targets) in enumerate(training_data_generator):
            # Move tensors to the right device
            print(f"processing input and target {i}")
            inputs = inputs.float().to(device)  # Convert inputs to Float tensors
            targets = targets.float().to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Create a mask for the padded regions of the targets
            mask = torch.sum(targets, dim=2) != 0  # Assuming the padding value is 0 across all features
            
            # Apply the mask to the outputs and targets
            outputs = outputs[mask]
            targets = targets[mask]
            
            # Compute loss only on non-padded parts
            loss = criterion(outputs, targets)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f'Loss: {loss.item()}')
            
        print(f'Epoch {epoch}, Loss: {loss.item()}')

    torch.save(model.state_dict(), 'model_weights.pth')
