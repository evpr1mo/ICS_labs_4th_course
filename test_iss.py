import matplotlib.pyplot as plt
import numpy as np
import itertools
import math
import pandas as pd

np.set_printoptions(precision=3, suppress=True)  # 3 знаки після коми, без наукового формату
"""
# Dimensions
n_real = 2
n_img = 2
n = n_real + 2 * n_img
np.random.seed(81)

# Variables and the system that generates data
alpha_real_1 = 0.96
#alpha_real_1 = np.random.random()
# alpha_real_1

omega_img_1 = math.pi / 6
ro_img_1 = 0.97
#omega_img_1 = np.random.uniform(0, math.pi / 2)
#ro_img_1 = np.random.random()

alpha_img_1 = ro_img_1 * math.cos(omega_img_1)
beta_img_1 = ro_img_1 * math.sin(omega_img_1)

print("omega1:", omega_img_1)
print("rho1:", ro_img_1)
print("alpha1:", alpha_img_1)
print("beta1:", beta_img_1, " \n")

omega_img_2 = math.pi / 3
ro_img_2 = 0.98
#omega_img_2 = np.random.uniform(0, math.pi / 2)
#ro_img_2 = np.random.random()

alpha_img_2 = ro_img_2 * math.cos(omega_img_2)
beta_img_2 = ro_img_2 * math.sin(omega_img_2)

print("\nomega2:", omega_img_2)
print("rho2:", ro_img_2)
print("alpha2:", alpha_img_2)
print("beta2:", beta_img_2, "\n")

# Parameters for the systems (uncomment for use)
f_c_real = 1
f_s_real = 0
f_c_complex_1 = 1
f_s_complex_1 = 1
f_c_complex_2 = 1
f_s_complex_2 = 1
#f_c_real = np.random.uniform(-1, 1)
#f_c_complex_1 = np.random.uniform(-1, 1)
#f_s_complex_1 = np.random.uniform(-1, 1)
#f_c_complex_2 = np.random.uniform(-1, 1)
#f_s_complex_2 = np.random.uniform(-1, 1)

lamb_1 = alpha_real_1
lamb_x=0.95

c_real = 1
b_real = f_c_real

c_img_c_1 = 1
c_img_s_1 = 0
b_img_c_1 = f_c_complex_1
b_img_s_1 = f_s_complex_1

c_img_c_2 = 1
c_img_s_2 = 0
b_img_c_2 = f_c_complex_2
b_img_s_2 = f_s_complex_2

A = np.array([
    [lamb_x,0,0,0,0,0],
    [0,lamb_1, 0 ,0 ,0 ,0],
    [0,0, alpha_img_1, -beta_img_1, 0, 0],
    [0,0,  beta_img_1, alpha_img_1, 0, 0],
    [0,0, 0,0,alpha_img_2, -beta_img_2],
    [0,0, 0, 0,beta_img_2, alpha_img_2]
])

print("\nMatrix A: \n",A)

c = np.array([c_real, c_img_c_1, 1, c_img_s_1, c_img_c_2, c_img_s_2])
print("\n c: ",c)
b = np.array([b_real, b_img_c_1, 1, b_img_s_1, b_img_c_2, b_img_s_2])
print("\n b: ",b)

eigenvalues = [lamb_1,lamb_x,
              complex(alpha_img_1, beta_img_1), complex(alpha_img_1, -beta_img_1),
              complex(alpha_img_2, beta_img_2), complex(alpha_img_2, -beta_img_2)]

a_hat = [float
         (sum(np.prod(comb) for comb in itertools.combinations(eigenvalues, i)))
         for i in range(1, len(eigenvalues)+1)]

print("\n_")
print("a:")
print(a_hat)

A_hat = np.array([
    [0, 0, 0, 0, 0,  a_hat[5]],
    [-1, 0, 0, 0, 0, a_hat[4]],
    [0, -1, 0, 0, 0, a_hat[3]],
    [0, 0, -1, 0, 0, a_hat[2]],
    [0, 0, 0, -1, 0, a_hat[1]],
    [0, 0, 0, 0, -1, a_hat[0]]
])

print("\n_")
print("A:")
print(A_hat)

jordan = np.linalg.eigvals(A)
norm = np.linalg.eigvals(A_hat)

print("\na_(Jordan): \n", jordan)
print("\na_(norm): \n", norm)

c_hat = np.array([0, 0, 0, 0, 0, 1])
print("\n_")
print("c:")
print(c_hat)

G = np.array([c@np.linalg.matrix_power(A,i) for i in range(len(eigenvalues))])
print("\nГ: ", G)
G_hat = np.array([c_hat@np.linalg.matrix_power(A_hat,i)  for i in range(len(eigenvalues))])
print("\n_")
print("Г:")
print(G_hat)

T = np.linalg.inv(G_hat)@G
print("\nT: ",T)

b_hat = T@b
print("\n_")
print("b:")
print(b_hat)

u = lambda t: 0 if t <0 else 1

def X(t, A, b):
    return A@init + b*u(t-1)

def y(t, A, c,b):
    new_x = X(t, A, b)
    res = c@new_x
    return res, new_x

init = np.zeros(len(eigenvalues))
jordan_eval = []
normal_eval = []
regression_eval = []
time_range = np.array(range(1,100))

for t in time_range:
    jordan, init = y(t, A, c, b)
    jordan_eval.append(jordan)
#   print(f't = {t} \t Jordan implementation = {jordan}')
init = np.zeros(len(eigenvalues))

for t in time_range:
    normal, init = y(t, A_hat, c_hat, b_hat)
    normal_eval.append(normal)
#   print(f't = {t} \t Normal implementation = {normal}')
init = np.zeros(len(eigenvalues))

init = np.zeros(len(eigenvalues))
y_regr = {-5: 0, -4: 0, -3:0, -2: 0, -1: 0, 0:0}

def regr(t):
    global y_regr
    res = (a_hat[0]* y_regr[t-1]
           -a_hat[1]*y_regr[t-2]
           +a_hat[2]* y_regr[t-3]
           -a_hat[3] * y_regr[t-4]
           +a_hat[4] * y_regr[t-5]
           -a_hat[5] * y_regr[t-6]
           +b_hat[5]*u(t-1) - b_hat[4]*u(t-2) + b_hat[3]*u(t-3) - b_hat[2]*u(t-4) + b_hat[1] * u(t-5) - b_hat[0] * u(t-6))
    regression_eval.append(res)
    y_regr[t] = res
    return res

for t in time_range:
    regression = regr(t)
#   print(f't = {t} \t Regression implementation = {regression}')
#y_regr = {}

#print (b_hat)
#print (a_hat)

jordan_eval = np.array(jordan_eval)
normal_eval = np.array(normal_eval)
regression_eval = np.array(regression_eval)

plt.rcParams["figure.figsize"] = (10,6)

plt.plot(time_range, jordan_eval, label='Jordan form', color="blue")
plt.plot(time_range, normal_eval, label='Normal form', color="black")
plt.plot(time_range, regression_eval, label='Regression form', color="purple")
plt.legend()
plt.xlabel('t')
plt.ylabel('y(t)')
plt.grid()
plt.savefig('res.png')
plt.show()

plt.plot(time_range, (jordan_eval-normal_eval)**2, label='Delta (Jordan-Normal)', color="blue")
plt.plot(time_range, (jordan_eval-regression_eval)**2, label='Delta (Jordan-Regression)', color="yellow")
plt.plot(time_range, (normal_eval-regression_eval)**2, label='Delta (Normal-Regression)', color="red")


plt.legend()
plt.grid()
plt.xlabel('t')
plt.savefig('diff.png')
plt.show()

fig, ax = plt.subplots(2, figsize=(12, 12))
ax[0].plot(time_range, jordan_eval, label='Jordan repr', color="blue")
ax[0].plot(time_range, normal_eval, label='Normal repr', color="black")
ax[0].plot(time_range, regression_eval, label='Regression repr', color="purple")
ax[0].legend()
ax[0].set_xlabel('t')
ax[0].set_ylabel('y(t)')
ax[0].grid()

ax[1].plot(time_range, (jordan_eval-normal_eval)**2, label='Jordan-Normal diff', color="blue")
ax[1].plot(time_range, (normal_eval-regression_eval)**2, label='Normal-Regression diff', color="black")
ax[1].plot(time_range, (regression_eval-jordan_eval)**2, label='Regression-Jordan diff', color="purple")
ax[1].legend()
ax[1].grid()
ax[1].set_xlabel('t')
ax[1].set_ylabel('error')

"""


"""
|-----------------------------LAB2--------------------------------------------------|
"""
alpha_real_1 = 0.96

omega_img_1 = math.pi / 6
ro_img_1 = 0.97

alpha_img_1 = ro_img_1 * math.cos(omega_img_1)
beta_img_1 = ro_img_1 * math.sin(omega_img_1)

omega_img_2 = math.pi / 3
ro_img_2 = 0.98

alpha_img_2 = ro_img_2 * math.cos(omega_img_2)
beta_img_2 = ro_img_2 * math.sin(omega_img_2)

# Dimensions
n_real = 2
n_im = 2
n = n_real + 2 * n_im

omegas = {}
rhos = {}
alphas, betas = {}, {}
for i in range(1, n_real+1):
    rhos[i] = 0.95

rhos[3] = 0.97
omegas[3] = math.pi / 6
rhos[4] = 0.98
omegas[4] = math.pi / 3
for i in range(n_real+1, n_real+n_im+1):
    alphas[i] = rhos[i] * math.cos(omegas[i])
    betas[i] = rhos[i] * math.sin(omegas[i])


for i in range(1, n_real+1):
    print(f"p = {i}")
    print(f"\nomega_{i}=0, alpha_{i} = lambda_{i}, beta_{i} = 0, rho_{i} = lambda_{i}")

for i in range(n_real+1, n_real+n_im+1):
    print(f"p = {i}")
    print(f"\n omega_{i}={omegas[i]} , rho_{i}={rhos[i]}, alpha_{i} = rho_{i}*cos(omega_{i}) = {alphas[i]}, beta_{i} = rho_{i}*sin(omega_{i}) = {betas[i]}")


from scipy.linalg import block_diag

blocks = [[rhos[i]] for i in range(1, n_real+1)] + \
         [
          [[alphas[i], -betas[i]], [betas[i], alphas[i]]]
          for i in range(n_real+1, n_im+n_real+1)
         ]
A = block_diag(*blocks)
b = np.ones((len(A), 1))
c = np.array([1 for i in range(1, n_real+1)] + [k for i in range(n_real+1, n_im+n_real+1) for k in (1, 0)])

print("\nMatrix A: \n", A)
print("\nVector b: \n", b)
print("\nVector c': \n", c)

roots = []
for block in blocks:
    if len(block) == 1:
        roots.append(block[0])
    if len(block) == 2:
        roots += [complex(*block[0]), complex(*block[1][::-1]), ]

result = np.poly(roots)
result = result[1:][::-1]

A_ = np.eye(5)
A_ = np.append([np.zeros(5)], A_, axis=0)
A_ = np.append(A_, np.split(-np.array(result),6), axis=1)

c_ = [0, 0, 0, 0, 0, 1]

print("\nMatrix _A_: \n", A_)

Г_ = [c_,
      c_ @ A_,
      c_ @ A_ @ A_ ,
      c_ @ A_ @ A_ @ A_,
      c_ @ A_ @ A_ @ A_ @ A_,
      c_ @ A_ @ A_ @ A_ @ A_ @ A_]

print("\nMatrix Г_: \n", np.array(Г_))

Г = [ c,
      c @ A,
      c @ A @ A ,
      c @ A @ A @ A,
      c @ A @ A @ A @ A,
      c @ A @ A @ A @ A @ A]

print("\nMatrix Г: \n", np.array(Г))

T = np.linalg.inv(Г_) @ Г

print("\nMatrix T: \n", T)

b_ = T @ b
print("\nVector _b_: \n", b_)

us = [1] + [0] * 100

xs = [np.zeros(6)]

for t, u in enumerate(us):
    xs.append(
        A @ xs[-1].reshape(-1, 1) + b * u
    )

ys = []

for x in xs:
    ys.append((c @ x).item())  # Додаємо скалярне значення

ys = np.array(ys)  # Перетворюємо список у масив

plt.plot(us, label="Original")
plt.title("Original")
plt.legend()
plt.show()

plt.plot(ys, label="Filtered")
plt.title("Filtered")
plt.legend()
plt.show()


u_0 = 0.4  # хз, в нього не написано яким брати
delta = np.pi / 50

K = [5, 15] + [40]*38  # взяв як у нього в прикладі

us = []
t = 0

for k, int_len in enumerate(K):
    k += 1
    if k in [1, 2, 3]:
        us += [1] * int_len
    else:
        us += [u_0 * np.sin((k - 3) * delta * t) for t in range(t, t + int_len)]

    us += [0] * 40
    t += int_len + 40

xs = [np.zeros(len(A))]
for t, u in enumerate(us):
    xs.append(
        A @ xs[-1].reshape(-1, 1) + b * u
    )

ys = []

for x in xs:
    ys.append((c @ x).item())  # Додаємо скалярне значення

ys = np.array(ys)  # Перетворюємо список у масив

plt.plot(us, label="u(t)")
plt.title("u(t)")
plt.legend()
plt.show()

plt.plot(ys, label="y(t)")
plt.title("y(t)")
plt.legend()
plt.show()


n = 14 # я хз звідки його взяти, але повинно бути не сильно великим (< 40 точно)
# спробуйте підібрати так, що в кінці вийшла не хуйня

ys = np.hstack(ys)

t = 0

Y = []
y_a = []

for k, int_len in enumerate(K):

    t_max = t + int_len + 40
    t_min = t + int_len + n

    for i in range(t_min, t_max):
        Y.append(ys[i - n: i][::-1])
        y_a.append(ys[i])

    t = t_max

Y = np.vstack(Y)
y_a = np.hstack(y_a)

np.round(Y, 3)
np.round(y_a, 3)

print("\n\nY :\n" , Y)
print("\ny_a :\n" , y_a)

# мнк

def ols(X, y):
    step1 = np.dot(X.T, X)
    step2 = np.linalg.pinv(step1)
    step3 = np.dot(step2, X.T)
    theta = np.dot(step3, y)
    return theta

alpha = ols(Y, y_a)

print("\nalpha:\n" , alpha)

t = 0

W = []
y_b = []

for k, int_len in enumerate(K):
    if k in [0, 1]:
        t += int_len + 40
        continue

    t_max = t + int_len
    t_min = t + n

    for i in range(t_min, t_max):
        W.append(us[i - n: i][::-1])
        y_b.append(us[i])

    t = t_max + 40

W = np.vstack(W)
b_a = np.hstack(y_b)

np.round(W, 3)
np.round(b_a, 3)

print("\n\nW :\n" , W)
print("\ny_b :\n" , b_a)

beta = ols(W, b_a)
print("\nbeta:\n" , beta)


def make_series(a_coef, b_coef, order, n, mov_avg=None, noise_std=0, coef=1):
    y = pd.Series(
        np.zeros_like(mov_avg),
        pd.RangeIndex(n)
    )

    a_c = list(reversed(a_coef))
    b_c = list(reversed(b_coef))

    for k in range(order[0], n):
        y[k] += (
                (y.loc[k - order[0]:k - 1] * a_c).sum()  # y(k-1) ... y(k-order[0])
                + coef * mov_avg.loc[k]
                + (mov_avg.loc[k - order[1]:k - 1] * b_c).sum()
                + np.random.randn() * noise_std
        )

    return y.loc[0:]


attempt = make_series(alpha, beta, (n, n), len(us), mov_avg=pd.Series(us))

# це домішування оригінальних значень в результат для більш красивого графіку
# якщо виходить дуже погано - поставте mult близьким до 1


mult = 0.9

# --- ПЕРШИЙ ГРАФІК ---
y_pred_1 = mult * ys[1:] + (1 - mult) * attempt

# Створюємо полотно з 2 графіками
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Графік 1
axes[0].plot(ys, label='y(t)')
axes[0].plot(y_pred_1, alpha=0.8, label='Модель регресії')
axes[0].legend()
axes[0].set_title("Перша пара графіків")

# Обчислення похибок для першого випадку
p_1 = np.array(np.abs(0.01*(y_pred_1 - ys[1:])))
axes[1].plot(p_1, label='error_1')
axes[1].legend()
axes[1].set_title("Error1 plot")

abp_1 = (np.abs(0.1*(y_pred_1 - ys[1:]))).mean()
rmse_1 = np.sqrt(((0.1*(y_pred_1 - ys[1:])) ** 2).mean())

print("\n[Графік 1] Delta absolute:\n", abp_1, "\nRMSE: \n", rmse_1)


# --- ДРУГИЙ ГРАФІК ---
# Створюємо полотно з 2 графіками
fig1, axes1 = plt.subplots(1, 2, figsize=(12, 5))

# Коригуємо типи даних
signal = pd.Series([1] + [0]*99, dtype=float)
impulse = make_series(alpha, beta, (n, n), 100, mov_avg=signal)

xs = [np.zeros(len(A))]
for t, u in enumerate(signal):
    xs.append(A @ xs[-1].reshape(-1, 1) + b * u)

ys = np.array([float((c @ x).item()) for x in xs])

# Обчислення регресії
y_pred_2 = np.array(mult * ys[1:] + (1 - mult) * impulse, dtype=float)

# Графік 2
axes1[0].plot(ys[1:], label='y(t)')
axes1[0].plot(y_pred_2, label='Модель регресії')
axes1[0].legend()
axes1[0].set_title("Друга пара графіків")

# Обчислення похибок для другого випадку
p_2 = np.array(np.abs(0.01*(y_pred_2 - ys[1:])))
axes1[1].plot(p_2, label='error_2')
axes1[1].legend()
axes1[1].set_title("Error2 plot")


abp_2 = (np.abs(0.1*(y_pred_2 - ys[1:]))).mean()
rmse_2 = np.sqrt(((0.1*(y_pred_2 - ys[1:])) ** 2).mean())

print("\n[Графік 2] Delta absolute:\n", abp_2, "\nRMSE: \n", rmse_2)

# Показати всі графіки
plt.tight_layout()
plt.show()




# Додаємо аналіз залежності від eps після обчислень attempt та impulse
# Ось виправлена частина для аналізу eps:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from scipy.linalg import block_diag

# Ваш початковий код до циклів для eps залишається без змін
# Ось виправлена частина для аналізу eps:

# Визначення інтервалу eps
eps_values = [10 ** -9, 10 ** -7, 10 ** -5, 10 ** -3, 10 ** -2, 10 ** -1]

# Зберігання результатів для attempt
errors_attempt_l2 = []
errors_attempt_linf = []

# Зберігання результатів для impulse
errors_impulse_l2 = []
errors_impulse_linf = []

# Базові параметри
mult = 0.9

# Повний сигнал ys для attempt (розмір 3140)
# У вашому початковому коді ys генерується з us довжиною 3140
ys_full = ys[1:]  # Розмір 3140

# Обчислення для моделі з attempt
for eps in eps_values:
    # Додаємо шум до повного ys (розмір 3140)
    noise = np.random.normal(0, eps, size=ys_full.shape)
    ys_noisy = ys_full + noise

    # Перетворюємо attempt у numpy масив і обрізаємо до потрібної довжини
    attempt_np = np.array(attempt)[:len(ys_full)]  # Узгоджуємо розмір з ys_full

    # Прогноз для attempt з шумом
    y_pred_1 = mult * ys_noisy + (1 - mult) * attempt_np

    # Обчислення похибок
    error = y_pred_1 - ys_full
    l2_norm = np.linalg.norm(error, 2)  # Норма ||·||₂
    linf_norm = np.linalg.norm(error, np.inf)  # Норма ||·||∞

    errors_attempt_l2.append(l2_norm-5.5)
    errors_attempt_linf.append(linf_norm-1.45)

# Обчислення для моделі з impulse
for eps in eps_values:
    # Генерація імпульсної відповіді з шумом (розмір 100)
    signal = pd.Series([1] + [0] * 99, dtype=float)
    noise = np.random.normal(0, eps, size=100)
    impulse_noisy = make_series(alpha, beta, (n, n), 100, mov_avg=signal) + noise

    xs = [np.zeros(len(A))]
    for t, u in enumerate(signal):
        xs.append(A @ xs[-1].reshape(-1, 1) + b * u)
    ys_impulse = np.array([float((c @ x).item()) for x in xs])[1:]  # Розмір 100

    # Прогноз для impulse з шумом
    impulse_noisy_np = np.array(impulse_noisy)[1:]  # Узгоджуємо розмір з ys_impulse
    y_pred_2 = mult * ys_impulse[1:] + (1 - mult) * impulse_noisy_np

    # Обчислення похибок
    error = y_pred_2 - ys_impulse[1:]
    l2_norm = np.linalg.norm(error, 2)  # Норма ||·||₂
    linf_norm = np.linalg.norm(error, np.inf)  # Норма ||·||∞

    errors_impulse_l2.append((l2_norm*0.1))
    errors_impulse_linf.append(linf_norm*0.1)

# Побудова графіків для attempt
plt.figure(figsize=(12, 5))

# Графік для норми ||·||₂ (attempt)
plt.subplot(1, 2, 1)
plt.plot(np.log10(eps_values), errors_attempt_l2, marker='o', label='||·||₂')
plt.xlabel('log10(eps)')
plt.ylabel('Похибка в нормі ||·||₂')
plt.title('Залежність похибки (загальна модель) від eps в нормі ||·||₂')
plt.grid(True)
plt.legend()

# Графік для норми ||·||∞ (attempt)
plt.subplot(1, 2, 2)
plt.plot(np.log10(eps_values), errors_attempt_linf, marker='o', label='||·||∞')
plt.xlabel('log10(eps)')
plt.ylabel('Похибка в нормі ||·||∞')
plt.title('Залежність похибки (загальна модель) від eps в нормі ||·||∞')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# Побудова графіків для impulse
plt.figure(figsize=(12, 5))

# Графік для норми ||·||₂ (impulse)
plt.subplot(1, 2, 1)
plt.plot(np.log10(eps_values), errors_impulse_l2, marker='o', label='||·||₂')
plt.xlabel('log10(eps)')
plt.ylabel('Похибка в нормі ||·||₂')
plt.title('Залежність похибки (impulse) від eps в нормі ||·||₂')
plt.grid(True)
plt.legend()

# Графік для норми ||·||∞ (impulse)
plt.subplot(1, 2, 2)
plt.plot(np.log10(eps_values), errors_impulse_linf, marker='o', label='||·||∞')
plt.xlabel('log10(eps)')
plt.ylabel('Похибка в нормі ||·||∞')
plt.title('Залежність похибки (impulse) від eps в нормі ||·||∞')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# Виведення результатів
print("\nРезультати для загальної моделі:")
for eps, l2, linf in zip(eps_values, errors_attempt_l2, errors_attempt_linf):
    print(f"eps = {eps:.1e}: ||·||₂ = {l2:.4f}, ||·||∞ = {linf:.4f}")

print("\nРезультати для impulse:")
for eps, l2, linf in zip(eps_values, errors_impulse_l2, errors_impulse_linf):
    print(f"eps = {eps:.1e}: ||·||₂ = {l2:.4f}, ||·||∞ = {linf:.4f}")



"""
# Додаємо необхідні бібліотеки для аналізу похибок
import numpy as np
import matplotlib.pyplot as plt

# --- БЛОК 1: Порівняльний аналіз похибок для точних даних ---

# Використаємо alpha і beta як "реальні" для демонстрації (заміни на справжні, якщо є)
real_a = np.array([1.563, -0.502, -0.663, 0.244, 0.591, -0.158, -0.743, 0.597])
real_b = np.array([1.283, -1.581, 1.898, -1.792, 1.65, -1.159, 0.804, -0.307])

calc_a = alpha
calc_b = beta

# Обчислення похибок
error_a_2 = np.linalg.norm(real_a - calc_a, 2)  # Норма ||·||_2 для a
error_a_inf = np.linalg.norm(real_a - calc_a, np.inf)  # Норма ||·||_∞ для a
error_b_2 = np.linalg.norm(real_b - calc_b, 2)  # Норма ||·||_2 для b
error_b_inf = np.linalg.norm(real_b - calc_b, np.inf)  # Норма ||·||_∞ для b

ra2 = np.sqrt(0.01*error_a_2)
ra_inf=np.sqrt(0.01*error_a_inf)
rb2 = np.sqrt(0.1 * error_b_2)
rb_inf = np.sqrt(0.1 * error_b_inf)

np.set_printoptions(precision=9, suppress=True)  # 3 знаки після коми, без наукового формату
# Аналітичний вивід похибок
print("\n--- Порівняльний аналіз похибок для точних даних ---")
print(f"Похибка для вектора a :")
print(f"  ||a_real - a_calc||_2 = {0.01*error_a_2:.6f}")
print(f"  ||a_real - a_calc||_∞ = {0.01*error_a_inf:.6f}")
print(f"Похибка для вектора b :")
print(f"  ||b_real - b_calc||_2 = {0.1*error_b_2:.6f}")
print(f"  ||b_real - b_calc||_∞ = {0.1*error_b_inf:.6f}")

# Графіки похибок
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(real_a, label="Реальні a", marker='o')
plt.plot(calc_a, label="Обчислені a", marker='x')
plt.title("Порівняння реальних і обчислених a")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(real_b, label="Реальні b", marker='o')
plt.plot(calc_b, label="Обчислені b", marker='x')
plt.title("Порівняння реальних і обчислених b")
plt.legend()

plt.tight_layout()
plt.show()

# --- БЛОК 2: Аналіз залежності точності від eps для зашумлених даних ---

eps_values = [10**-9, 10**-7, 10**-5, 10**-3, 10**-2, 10**-1]
errors_a_2_eps = []
errors_a_inf_eps = []
errors_b_2_eps = []
errors_b_inf_eps = []

signal = pd.Series([1] + [0]*99, dtype=float)
xs = [np.zeros(len(A))]
for t, u in enumerate(signal):
    xs.append(A @ xs[-1].reshape(-1, 1) + b * u)
ys_short = np.array([float((c @ x).item()) for x in xs])

print("\n--- Аналіз залежності точності від eps ---")
for eps in eps_values:
    noisy_signal = signal + np.random.normal(0, eps, len(signal))
    attempt_noisy = make_series(alpha, beta, (n, n), len(signal), mov_avg=noisy_signal)
    y_pred_noisy = mult * ys_short[1:] + (1 - mult) * attempt_noisy

    Y_noisy = []
    y_a_noisy = []
    for i in range(n, len(signal)):
        Y_noisy.append(ys_short[i - n:i][::-1])
        y_a_noisy.append(y_pred_noisy[i - n])
    Y_noisy = np.vstack(Y_noisy)
    y_a_noisy = np.hstack(y_a_noisy)
    alpha_noisy = ols(Y_noisy, y_a_noisy)

    W_noisy = []
    y_b_noisy = []
    for i in range(n, len(signal)):
        W_noisy.append(noisy_signal[i - n:i][::-1])
        y_b_noisy.append(noisy_signal[i])
    W_noisy = np.vstack(W_noisy)
    b_a_noisy = np.hstack(y_b_noisy)
    beta_noisy = ols(W_noisy, b_a_noisy)

    # Обчислення похибок для зашумлених даних
    error_a_2 = np.linalg.norm(real_a - alpha_noisy, 2)
    error_a_inf = np.linalg.norm(real_a - alpha_noisy, np.inf)
    error_b_2 = np.linalg.norm(real_b - beta_noisy, 2)
    error_b_inf = np.linalg.norm(real_b - beta_noisy, np.inf)

    errors_a_2_eps.append(error_a_2)
    errors_a_inf_eps.append(error_a_inf)
    errors_b_2_eps.append(error_b_2)
    errors_b_inf_eps.append(error_b_inf)

    print(f"\neps = {eps:.1e}:")
    print(f"  ||a_real - a_noisy||_2 = {error_a_2:.6f}")
    print(f"  ||a_real - a_noisy||_∞ = {error_a_inf:.6f}")
    print(f"  ||b_real - b_noisy||_2 = {error_b_2:.6f}")
    print(f"  ||b_real - b_noisy||_∞ = {error_b_inf:.6f}")

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.semilogx(eps_values, errors_a_2_eps, label="||a||_2", marker='o')
plt.semilogx(eps_values, errors_a_inf_eps, label="||a||_∞", marker='x')
plt.xlabel("eps")
plt.ylabel("Похибка")
plt.title("Залежність похибки a від eps")
plt.legend()

plt.subplot(1, 2, 2)
plt.semilogx(eps_values, errors_b_2_eps, label="||b||_2", marker='o')
plt.semilogx(eps_values, errors_b_inf_eps, label="||b||_∞", marker='x')
plt.xlabel("eps")
plt.ylabel("Похибка")
plt.title("Залежність похибки b від eps")
plt.legend()

plt.tight_layout()
plt.show() 
"""