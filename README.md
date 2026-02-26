# Develop a Convolutional Deep Neural Network for Image Classification

## AIM
To develop a convolutional deep neural network (CNN) for image classification and to verify the response for new images.

##   PROBLEM STATEMENT AND DATASET
Include the Problem Statement and Dataset.

## Neural Network Model
<img width="998" height="698" alt="image" src="https://github.com/user-attachments/assets/979e7f8b-c93b-486c-9bbb-9071def832db" />


## DESIGN STEPS
### STEP 1: 

Import the required libraries (torch, torchvision, torch.nn, torch.optim) and load the image dataset with necessary preprocessing like normalization and transformation.
### STEP 2: 

Split the dataset into training and testing sets and create DataLoader objects to feed images in batches to the CNN model.

### STEP 3: 

Define the CNN architecture using convolutional layers, ReLU activation, max pooling layers, and fully connected layers as implemented in the CNNClassifier class.

### STEP 4: 

Initialize the model, define the loss function (CrossEntropyLoss), and choose the optimizer (Adam) for training the network.

### STEP 5: 

Train the model using the training dataset by performing forward pass, computing loss, backpropagation, and updating weights for multiple epochs.

### STEP 6: 
Evaluate the trained model on test images and verify the classification accuracy for new unseen images.




## PROGRAM

### Name: VENKATESAN R

### Register Number: 212224230299

```python
class CNNClassifier(nn.Module):
    def __init__(self, input_size):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1)
        self.conv3 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.fc1 = nn.Linear(128*3*3,128)
        self.fc2 = nn.Linear(128,64)
        self.fc3 = nn.Linear(64,10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x=x.view(x.size(0),-1)
        x=torch.relu(self.fc1(x))
        x=torch.relu(self.fc2(x))
        x=self.fc3(x)
        return x



# Initialize model, loss function, and optimizer
model = CNNClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the Model
def train_model(model, train_loader, num_epochs=3):

    for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
      optimizer.zero_grad()
      outputs = model(images)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()
      running_loss += loss.item()e

        
        
        
        print('Name: VENKATESAN R')
        print('Register Number:212224230299')
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

```

### OUTPUT

## Training Loss per Epoch

<img width="331" height="203" alt="{2ADB6676-CD27-4ACB-BAE8-6C79E39B7C33}" src="https://github.com/user-attachments/assets/d97370fc-0ba7-4a58-a695-0d7a36e7fbe0" />


## Confusion Matrix

<img width="711" height="620" alt="{EA4F18F4-864B-4FF5-8AB3-B26022786A26}" src="https://github.com/user-attachments/assets/d0276795-e2f1-4d6e-b4a0-7e04a1c1e831" />


## Classification Report
<img width="478" height="321" alt="{FEC99D75-9428-433D-BC7A-54F85E23975D}" src="https://github.com/user-attachments/assets/01eb4eaf-7c4e-44b6-ab3a-2331ca4b15d4" />

### New Sample Data Prediction
<img width="494" height="561" alt="{876E264D-7F69-419C-8FE7-FEE744B39DF4}" src="https://github.com/user-attachments/assets/9724de64-929d-452f-81f8-bd9b81dabb9e" />


## RESULT
The program Exucuted Successfully.
