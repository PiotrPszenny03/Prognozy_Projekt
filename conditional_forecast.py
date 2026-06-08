import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt
# 1. Wczytanie danych
df = pd.read_excel("Prognozy_Baza_Danych.xlsx", sheet_name=0)
df['okres'] = pd.to_datetime(df['Zmienna'].str.replace(' M', '-'), format='%Y-%m')
df['okres'] = df['okres'] + pd.offsets.MonthEnd(0)
df['miesiąc'] = df['okres'].dt.month
df['Wakacje'] = df['miesiąc'].isin([7, 8]).astype(int)
df['Ferie'] = df['miesiąc'].isin([1, 2]).astype(int)
df = df.sort_values('okres').reset_index(drop=True)
    
# 2. Podział na zbiór uczący i testowy
mask_train = (df['okres'] >= '2015-02-01') & (df['okres'] <= '2018-05-31')
mask_test = (df['okres'] > '2018-05-31') & (df['okres'] <= '2018-08-31')
train_data = df[mask_train].copy().dropna(subset=['Wynagrodzenia'])
test_data = df[mask_test].copy()
# 3. Modelowanie Pogody (Holt-Winters) na zbiorze uczącym
ts_train_weather = train_data['Pogoda'].values
hw_model = ExponentialSmoothing(ts_train_weather, seasonal_periods=12, trend='add', seasonal='add', initialization_method="estimated")
hw_fit = hw_model.fit()
weather_forecast = hw_fit.forecast(3)
# 4. Model OLS (Turyści)
y_train = np.log(train_data['Turyści'].values)
X_train = pd.DataFrame()
X_train['Pogoda'] = train_data['Pogoda']
X_train['Wakacje'] = train_data['Wakacje']
X_train['Ferie'] = train_data['Ferie']
X_train['Wynagrodzenia'] = train_data['Wynagrodzenia']
X_train.insert(0, 'const', 1.0)
ols_model = sm.OLS(y_train, X_train).fit()
# 5. Prognoza warunkowa na 3 miesiące (Czerwiec, Lipiec, Sierpień 2018)
# Wartości zaprognozowane (od użytkownika, ex-ante)
wynagrodzenia_forecast = [4715.33, 4733.88, 4752.42]
wakacje_forecast = test_data['Wakacje'].values
ferie_forecast = test_data['Ferie'].values
X_test = pd.DataFrame()
X_test['Pogoda'] = weather_forecast
X_test['Wakacje'] = wakacje_forecast
X_test['Ferie'] = ferie_forecast
X_test['Wynagrodzenia'] = wynagrodzenia_forecast
X_test.insert(0, 'const', 1.0)
# Prognoza w logarytmach i odlogarytmowanie (wartości absolutne)
y_pred_log = ols_model.predict(X_test)
y_pred = np.exp(y_pred_log)
# 6. Błędy ex-post na zbiorze uczącym (dopasowanie modelu)
y_train_true = train_data['Turyści'].values
y_train_pred = np.exp(ols_model.fittedvalues)
def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
rmse_ex_post_train = np.sqrt(np.sum(np.square(y_train_true - y_train_pred)) / len(y_train_true))
mape_ex_post_train = mape(y_train_true, y_train_pred)
rmspe_ex_post_train = (rmse_ex_post_train / np.mean(y_train_true)) * 100
# 7. Błędy ex-post prognozy wygasłej (na zbiorze testowym - po poznaniu rzeczywistości)
y_true = test_data['Turyści'].values
rmse_ex_post_test = np.sqrt(np.sum(np.square(y_true - y_pred)) / len(y_true))
mape_ex_post_test = mape(y_true, y_pred)
rmspe_ex_post_test = (rmse_ex_post_test / np.mean(y_true)) * 100
# 8. Błędy ex-ante prognozy (teoretyczne, liczone przed faktem)
pred_results = ols_model.get_prediction(X_test)
se_obs_log = pred_results.se_obs # Błąd ex-ante dla modelu logarytmicznego ln(Y)
# Ze względu na postać logarytmiczną modelu, błąd bezwzględny dla Y aproksymujemy jako:
s_p_ex_ante = se_obs_log * y_pred
# A błąd względny V_p to bezpośrednio odchylenie standardowe z modelu logarytmicznego w %:
v_p_ex_ante = se_obs_log * 100
print("\n--- Podsumowanie zapisano do pliku prognoza_ex_ante_summary.txt ---")
with open("prognoza_ex_ante_summary.txt", "w", encoding="utf-8") as f:
    f.write("--- Wyniki prognozy warunkowej (Turyści) ---\n")
    f.write("Okres: Czerwiec - Sierpień 2018\n\n")
    for i, (d, p, r) in enumerate(zip(test_data['okres'].dt.strftime('%Y-%m'), y_pred, y_true)):
        f.write(f"{d} - Prognoza: {p:.0f}, Rzeczywiste: {r}\n")
        f.write(f"    -> Teoretyczny błąd ex-ante: {s_p_ex_ante[i]:.0f} osób ({v_p_ex_ante[i]:.2f}%)\n")
    
    f.write("\n--- Podsumowanie jakości ---\n")
    f.write("\n1. Błędy dopasowania ex-post (na zbiorze uczącym):\n")
    f.write(f"RMSE:  {rmse_ex_post_train:.2f}\n")
    f.write(f"MAPE:  {mape_ex_post_train:.2f}%\n")
    f.write(f"RMSPE: {rmspe_ex_post_train:.2f}%\n")
    f.write("\n2. Błędy ex-post prognozy wygasłej (porównanie z rzeczywistością dla prognozy): \n")
    f.write(f"RMSE:  {rmse_ex_post_test:.2f}\n")
    f.write(f"MAPE:  {mape_ex_post_test:.2f}%\n")
    f.write(f"RMSPE: {rmspe_ex_post_test:.2f}%\n")
print("--- Wyniki prognozy warunkowej i błędy teoretyczne ex-ante ---")
for i, (d, p, r) in enumerate(zip(test_data['okres'].dt.strftime('%Y-%m'), y_pred, y_true)):
    print(f"{d} - Prognoza: {p:.0f} (Ex-ante: +/- {s_p_ex_ante[i]:.0f} osób, {v_p_ex_ante[i]:.2f}%), Rzeczywiste: {r}")
print("\n--- Błędy ex-post prognozy wygasłej (rzeczywiste odchylenia) ---")
print(f"RMSE:  {rmse_ex_post_test:.2f}")
print(f"MAPE:  {mape_ex_post_test:.2f}%")
print(f"RMSPE: {rmspe_ex_post_test:.2f}%")
# 9. Wykres prognozy
plot_dates_train = train_data['okres'].dt.to_period('M').dt.to_timestamp()
plot_dates_test = test_data['okres'].dt.to_period('M').dt.to_timestamp()
plt.figure(figsize=(10, 6))
mask_plot = train_data['okres'] >= '2017-01-01'
plt.plot(plot_dates_train[mask_plot], np.exp(y_train[mask_plot]), label="Wartości rzeczywiste (Trening)", marker='o', markersize=5, color='#1f77b4')
plt.plot(plot_dates_test, y_true, label="Wartości rzeczywiste (Test)", marker='o', markersize=5, color='#2ca02c')
# Dodanie przedziałów ufności na podstawie błędu ex-ante
plt.fill_between(plot_dates_test, y_pred - s_p_ex_ante, y_pred + s_p_ex_ante, color='#d62728', alpha=0.2, label='Błąd ex-ante (+/- 1 odch. std)')
plt.plot(plot_dates_test, y_pred, label="Prognoza warunkowa", linestyle='--', marker='X', markersize=7, color='#d62728')
plt.ticklabel_format(style='plain', axis='y')
plt.title("Prognoza warunkowa liczby turystów na miesiące VI-VIII 2018")
plt.xlabel("Data")
plt.ylabel("Liczba turystów")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("wykres_prognoza_warunkowa.png")
print("\nWykres zapisano jako wykres_prognoza_warunkowa.png")