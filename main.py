import os
import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
from scipy import stats

# Середнє значення
def get_mean (values):
    return np.mean(values)

# Медіана
def get_median (values):
    return np.median(values)

# Дисперсія
def get_variance (values):
    return np.var(values, ddof=1) # для незміщеної оцінки дисперсії

# Стандартне відхилення
def get_standard_deviation (values):
    return np.sqrt(get_variance (values))

# Довірчий інтервал
def confidence_interval_mean (values, mean, std, alpha = 0.05):
    n_ = len(values)
    t_crit = stats.t.ppf(1 - alpha/2, n_ - 1)
    margin_error = t_crit * std / np.sqrt(n_)
    ci_mean = (mean - margin_error, mean + margin_error)
    return ci_mean

# !!! Розмір файлу, що буде завантажено 653,3 МБ. Рекомендацію про не більше 10 000 записів уже прочитав коли робота з кодом завершена на 90%.
path = kagglehub.dataset_download("aryan208/financial-transactions-dataset-for-fraud-detection") # Data Set
csv_file = os.path.join(path, os.listdir(path)[0]) # Шлях до файлу

# Завантажуємо дані у DataFrame
data = pd.read_csv(csv_file)
print(f"\nПерші 5 рядків даних:\n{data.head()}")

"""
Рішення наступного блоку зроблено за допомогою AI.
Метод data.info() стандартно виводить інформацію безпосередньо в консоль і повертає None.
Щоб вбудувати інформацію про DataFrame в вивід нашого текстового рядка,
потрібно спочатку зберегти цей вивід у змінну.
Зробимо це за допомогою об'єкта StringIO з модуля io.
"""
buf = StringIO()
data.info(buf=buf)
info_str = buf.getvalue()

print(f"\nІнформація про DataFrame:\n {info_str}")

# Перевіряємо на наявність пропущених значень і заповнення їх середнім (якщо є)
null_count = data['amount'].isnull().sum()

if null_count > 0:
    mean_value = data['amount'].mean()
    filled_amount = data['amount'].fillna(mean_value)
    data['amount'] = filled_amount
    print("В колонці 'amount' не залишається пропущених значень, всі вони замінені на середнє значення по цьому стовпцю")
else:
    print('Пропущенні значення в колонці "amount" відсутні')

# Обчислюємо основні характеристики
amount = data['amount']

mean_val = get_mean (amount)
median_val = get_median (amount)
variance_val = get_variance (amount)
std_val = get_standard_deviation (amount)
mode_val = data['amount'].mode()[0]

print(f"\nСтатистичні характеристики для колонки 'amount':"
      f"\nСереднє: {mean_val}"
      f"\nМедіана: {median_val}"
      f"\nМода: {mode_val}"
      f"\nДисперсія: {variance_val}"
      f"\nСтандартне відхилення: {std_val}")

# Гістограма розподілу 'amount'
plt.figure(figsize=(8, 5))
sns.histplot(amount, bins=30, kde=True, color='green')
plt.title("Гістограма розподілу значень 'amount'")
plt.xlabel("Amount")
plt.ylabel("Частота")
plt.tight_layout()
hist_filename = "histogram_amount.png"
plt.savefig(hist_filename)
plt.grid(axis='y')
plt.show()

# Діаграма розсіювання
plt.figure(figsize=(12, 7))
sns.scatterplot(x='amount', y='velocity_score', data=data, color='orange')
plt.title("Діаграма розсіювання: amount vs velocity score")
plt.xlabel("Amount")
plt.ylabel("Velocity score")
plt.tight_layout()
scatter_filename = "scatter_amount_velocity_score.png"
plt.savefig(scatter_filename)
plt.show()

ci = tuple(float(x) for x in confidence_interval_mean (amount, mean_val, std_val))
print(f"\n95% довірчий інтервал для середнього 'amount', верхня і нижня межа: {ci}")

stats_df = pd.DataFrame({
    "Показник": ["Середнє", "Медіана", "Мода", "Дисперсія", "Стандартне відхилення", "CI 95% Довірчий інтервал - нижня межа", "CI 95% Довірчий інтервал - верхня межа"],
    "Значення": [mean_val, median_val, mode_val, variance_val, std_val, {ci[0]}, {ci[1]}]
})

excel_filename = "amount_statistics.xlsx"
with pd.ExcelWriter(excel_filename, engine='xlsxwriter') as writer:
    stats_df.to_excel(writer, sheet_name='Статистика', index=False)

    # Аркуш для графіків
    workbook = writer.book
    worksheet = workbook.add_worksheet("Графіки")
    # Додавання графіків
    worksheet.insert_image("B2", hist_filename)
    worksheet.insert_image("B30", scatter_filename)

print("\nСтатистичні показники та графіки експортовано в Excel-файл:", excel_filename)

# Перевірка на нормальність
stat, p = stats.shapiro(data['amount'])
alpha = 0.05
if p > alpha:
    print("Розподіл 'amount' нормальний (не відхиляємо нульову гіпотезу).")
else:
    print("Розподіл 'amount' відрізняється від нормального (відхиляємо нульову гіпотезу).")

# Коефіцієнт кореляції Пірсона
correlation, p_value = stats.pearsonr(data['amount'], data['velocity_score'])
print(f"Коефіцієнт кореляції між 'amount' та 'velocity_score': {correlation:.4f}")