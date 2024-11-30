import torch
import torch.nn as nn
import torch.nn.functional as F

class ChatbotNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.2):
        super(ChatbotNN, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        self.batch_norm1 = nn.BatchNorm1d(hidden_size)
        self.batch_norm2 = nn.BatchNorm1d(hidden_size)
        
    def forward(self, x):
        out = self.l1(x)
        out = self.batch_norm1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.l2(out)
        out = self.batch_norm2(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.l3(out)
        return out

class ChatbotModel:
    def __init__(self, input_size, hidden_size=16, output_size=None, learning_rate=0.001):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size if output_size is not None else input_size
        self.learning_rate = learning_rate
        
        # Inicializar el modelo
        self.model = ChatbotNN(self.input_size, self.hidden_size, self.output_size)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
    def train(self, X_train, y_train, epochs=1000):
        # Convertir datos a tensores de PyTorch
        X = torch.FloatTensor(X_train)
        y = torch.LongTensor(y_train)
        
        # Entrenamiento
        self.model.train()
        for epoch in range(epochs):
            # Forward pass
            outputs = self.model(X)
            loss = self.criterion(outputs, y)
            
            # Backward pass y optimización
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            if (epoch + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
                
    def predict(self, X):
        # Convertir entrada a tensor
        X = torch.FloatTensor(X)
        
        # Cambiar a modo evaluación
        self.model.eval()
        
        # Desactivar el cálculo de gradientes
        with torch.no_grad():
            outputs = self.model(X)
            # Aplicar softmax para obtener probabilidades
            probabilities = F.softmax(outputs, dim=1)
            
            # Obtener el índice de la clase con mayor probabilidad y su valor
            confidence, predicted = torch.max(probabilities, 1)
            
        return predicted.numpy(), confidence.item()
    
    def save_model(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'output_size': self.output_size,
        }, path)
    
    @classmethod
    def load_model(cls, path):
        # Cargar el checkpoint
        checkpoint = torch.load(path)
        
        # Crear una nueva instancia del modelo
        model = cls(
            input_size=checkpoint['input_size'],
            hidden_size=checkpoint['hidden_size'],
            output_size=checkpoint['output_size']
        )
        
        # Cargar los estados del modelo y optimizador
        model.model.load_state_dict(checkpoint['model_state_dict'])
        model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return model
