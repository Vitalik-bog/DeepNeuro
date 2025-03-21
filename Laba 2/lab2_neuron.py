import pandas as pd  # библиотека pandas нужна для работы с данными
import matplotlib.pyplot as plt  # matplotlib для построения графиков
import numpy as np  # numpy для работы с векторами и матрицами

# Считываем данные
df = pd.read_csv('data.csv')

# Смотрим первые строки данных
print(df.head())

# Три столбца - это признаки, четвертый - целевая переменная
y = df.iloc[:, 4].values  # Целевая переменная

# Преобразуем строковые значения в числовые
y = np.where(y == "Iris-setosa", 1, -1)

# Возьмем три признака для удобства работы
X = df.iloc[:, [0, 1, 2]].values  # Первые три признака

# Функция нейрона (обновлена для трех признаков)
def neuron(w, x):
    if (w[1] * x[0] + w[2] * x[1] + w[3] * x[2] + w[0]) >= 0:
        predict = 1
    else:
        predict = -1
    return predict

# Проверим как работает нейрон (веса зададим произвольно)
w = np.array([0, 0.1, 0.4, 0.2])  # Добавили третий вес для третьего признака
eta = 0.01  # Скорость обучения
w_iter = []  # Список для сохранения весов на каждой итерации

# Процедура обучения (обновлена для трех признаков)
for xi, target, j in zip(X, y, range(X.shape[0])):
    predict = neuron(w, xi)
    w[1:] += (eta * (target - predict)) * xi  # Корректировка весов для трех признаков
    w[0] += eta * (target - predict)  # Корректировка свободного члена
    if j % 10 == 0:  # Каждую 10-ю итерацию сохраняем веса
        w_iter.append(w.tolist())

# Подсчет ошибок
sum_err = 0
for xi, target in zip(X, y):
    predict = neuron(w, xi)
    sum_err += (target - predict) / 2

print("Всего ошибок: ", sum_err)

# Визуализация (убрана, так как мы работаем с тремя признаками)