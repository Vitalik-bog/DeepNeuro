import random # импортируем генератор случайных чисел

random_list = [random.randint(1, 100) for _ in range(10)]#генерация рандомного списка от 1 до 100
print("Созданный список:", random_list)
even_sum = 0  # Переменная для хранения суммы четных чисел
for number in random_list:
    if number % 2 == 0:  # Проверяем, является ли число четным
        even_sum += number  # Если да, добавляем его к общей сумме
print("Полученная сумма чисел", even_sum)