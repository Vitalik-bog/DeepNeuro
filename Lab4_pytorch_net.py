import pandas as pd
import torch
import torch.nn as nn

# Значение n из ЭИОС
n = 7

print('Решите задачу предсказания дохода по возрасту.')

# Загрузка данных
df = pd.read_csv('dataset_simple.csv')
X = df.iloc[:, 0].values.reshape(-1, 1)  # Первый столбец: возраст
y = df.iloc[:, 1].values.reshape(-1, 1)  # Второй столбец: доход

# Нормализация данных 
mean_X = X.mean(axis=0)  # Среднее значение по каждому признаку
std_X = X.std(axis=0)    # Стандартное отклонение по каждому признаку
X = (X - mean_X) / std_X  # Стандартизация данных
X = torch.Tensor(X)       # Преобразование в тензор PyTorch
y = torch.Tensor(y)       # Преобразование меток в тензор PyTorch

# Определение архитектуры нейронной сети для регрессии
class NNetRegression(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        super(NNetRegression, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_size)
        )
    
    def forward(self, X):
        return self.layers(X)

# Параметры сети
input_size = X.shape[1]  # Размерность входных данных (1 признак: возраст)
hidden_size = 5          # Число нейронов в скрытом слое
output_size = 1          # Один выходной нейрон для регрессии

# Создание экземпляра сети
net = NNetRegression(input_size, hidden_size, output_size)

# Функция потерь и оптимизатор
loss_fn = nn.MSELoss()  # Среднеквадратичная ошибка для задачи регрессии
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

# Обучение модели
epochs = 100
for epoch in range(epochs):
    pred = net(X)  # Прогнозы модели
    loss = loss_fn(pred, y)  # Вычисление функции потерь
    optimizer.zero_grad()  # Обнуление градиентов
    loss.backward()        # Вычисление градиентов
    optimizer.step()       # Обновление параметров модели
    
    if (epoch + 1) % 10 == 0:
        print(f'Ошибка на эпохе {epoch + 1}: {loss.item()}')

# Оценка модели
with torch.no_grad():
    pred = net(X)  # Прогнозы модели на тестовых данных
    mae = torch.mean(torch.abs(y - pred)).item()  # Средняя абсолютная ошибка
    print(f'Ошибка (MAE): {mae:.2f}')