import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import Dataset
from model import Model
import wandb

wandb.init(project='SDETENN')

num_classes = 3  # each output
batch_size = 32
learning_rate = 0.01
num_epochs = 100

train_dataset = Dataset(data_dir='dataset/')
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = Dataset(data_dir='test/')
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

model = Model(num_classes)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.is_available())
model = model.to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    model.train()
    running_train_loss = 0.0
    for images, rotations, l2_values, r2_values in train_loader:
        images = images.to(device).float()
        rotations = rotations.to(device).float()
        l2_values = l2_values.to(device).float()
        r2_values = r2_values.to(device).float()

        optimizer.zero_grad()
        outputs = model(images)
        rotation_pred, l2_pred, r2_pred = outputs[:, 0], outputs[:, 1], outputs[:, 2]

        loss = criterion(rotation_pred, rotations) + criterion(l2_pred, l2_values) + criterion(r2_pred, r2_values)
        loss.backward()
        optimizer.step()

        running_train_loss += loss.item() * images.size(0)
        wandb.log({"Train Loss": loss.item()})


    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for val_images, val_rotations, val_l2_values, val_r2_values in val_loader:
            val_images = val_images.to(device).float()
            val_rotations = val_rotations.to(device).float()
            val_l2_values = val_l2_values.to(device).float()
            val_r2_values = val_r2_values.to(device).float()

            val_outputs = model(val_images)
            val_rotation_pred, val_l2_pred, val_r2_pred = val_outputs[:, 0], val_outputs[:, 1], val_outputs[:, 2]

            val_loss = criterion(val_rotation_pred, val_rotations) + criterion(val_l2_pred, val_l2_values) + criterion(val_r2_pred, val_r2_values)
            running_val_loss += val_loss.item() * val_images.size(0)

    epoch_val_loss = running_val_loss / len(val_dataset)
    print(f'Validation - Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_val_loss:.4f}')
    
    # Log validation loss to wandb
    wandb.log({"Validation Loss": epoch_val_loss})

    torch.save(model.state_dict(), f'models/model_loss{epoch_val_loss}_e{epoch+1}.pth')
