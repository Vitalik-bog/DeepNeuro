import torch

# 1. Создайте тензор x целочисленного типа, хранящий случайное значение
x = torch.randint(1, 10, (1,), dtype=torch.int32)  # Случайное целое число от 1 до 10
print(f"Исходный тензор x: {x}")

# 2. Преобразуйте тензор к типу float32
x = x.to(dtype=torch.float32)
print(f"Тензор x после преобразования: {x}")

# 3. Проведите операции
# Определяем n (для примера возьмём n = 3 тк мой номер четный) 
n = 3  

# a) Возведение в степень n
result = x ** n
print(f"Результат возведения в степень {n}: {result}")

# b) Умножение на случайное значение в диапазоне [1, 10]
random_value = torch.rand(1) * 9 + 1  # Случайное число от 1 до 10
result = result * random_value
print(f"Результат умножения на {random_value.item()}: {result}")

# c) Взятие экспоненты
result = torch.exp(result)
print(f"Результат взятия экспоненты: {result}")

# 4. Вычисление производной
# Для вычисления производной нужно установить requires_grad=True
x.requires_grad_(True)
result = torch.exp((x ** n) * random_value)  # Пересчитываем результат с градиентами
result.backward()  # Вычисляем градиенты
print(f"Производная d(result)/dx: {x.grad}")

#2задача
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Загрузка данных о цветках ириса
data = load_iris()
X = data.data  # Признаки
y = data.target  # Целевые метки

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Нормализация данных
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Преобразование данных в тензоры PyTorch
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# Определение модели (один полносвязный слой)
model = nn.Sequential(
    nn.Linear(4, 16),  # 4 признака -> 16 нейронов
    nn.ReLU(),         # Функция активации ReLU
    nn.Linear(16, 3)   # 16 нейронов -> 3 класса
)

# Функция потерь и оптимизатор
loss_fn = nn.CrossEntropyLoss()  # Для многоклассовой классификации
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Обучение модели
epochs = 100
for epoch in range(epochs):
    # Прямой проход
    y_pred = model(X_train)
    loss = loss_fn(y_pred, y_train)
    
    # Обратный проход
    optimizer.zero_grad()  # Обнуляем градиенты
    loss.backward()        # Вычисляем градиенты
    optimizer.step()       # Обновляем веса
    
    if (epoch + 1) % 10 == 0:
        print(f"Эпоха [{epoch+1}/{epochs}], Потери: {loss.item():.4f}")

# Тестирование модели
with torch.no_grad():
    y_test_pred = model(X_test)
    _, predicted = torch.max(y_test_pred, 1)
    accuracy = (predicted == y_test).sum().item() / len(y_test) * 100
    print(f"Точность на тестовых данных: {accuracy:.2f}%")
