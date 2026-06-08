import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
# Wczytanie danych
df = pd.read_excel("Prognozy_Baza_Danych.xlsx", sheet_name=0)
# create 'okres' 
df['okres'] = pd.to_datetime(df['Zmienna'].str.replace(' M', '-'), format='%Y-%m')
df['okres'] = df['okres'] + pd.offsets.MonthEnd(0)
df = df.sort_values('okres').reset_index(drop=True)
# Definiujemy okresy (zgodnie z "ten sam co dla calego modelu")
# Model OLS byl na: 2015-02-01 do 2018-05-31
mask_train = (df['okres'] >= '2015-02-01') & (df['okres'] <= '2018-05-31')
train_data = df[mask_train].copy()
# Ex-post test na nastepne 3 miesiace (czerwiec, lipiec, sierpien 2018)
mask_test = (df['okres'] > '2018-05-31') & (df['okres'] <= '2018-08-31')
test_data = df[mask_test].copy()
# Przygotowanie szeregu
ts_train = train_data['Pogoda'].values
# Model Holt-Winters
# Poniewaz pogoda moze miec wartosci ujemne (np. w lutym), musimy uzyc modelu addytywnego
hw_model = ExponentialSmoothing(ts_train, seasonal_periods=12, trend='add', seasonal='add', initialization_method="estimated")
hw_fit = hw_model.fit()
# Wartości dopasowane (in-sample) dla okresu modelu
fitted = hw_fit.fittedvalues
# Ewaluacja bledu ex-post na probie modelu
def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
# RMSE = Pierwiastek(suma kwadratów błedów prognoz wygasłych / liczba prognoz wygasłych)
rmse_val = np.sqrt(np.sum(np.square(ts_train - fitted)) / len(ts_train))
mape_val = mape(ts_train, fitted)
# RMSPE = RMSE / Średnia z wartości rzeczywistych
rmspe_val = (rmse_val / np.mean(ts_train)) * 100
# 5. Prognoza na 3 miesiące (ex-ante)
weather_forecast = hw_fit.forecast(3)
# Symulacja do uzyskania teoretycznego błędu ex-ante (ponieważ HW w statsmodels nie daje analitycznego S_p)
np.random.seed(42)
simulations = hw_fit.simulate(nsimulations=3, repetitions=1000, error="add")
s_p_ex_ante = simulations.std(axis=1)
v_p_ex_ante = (s_p_ex_ante / weather_forecast) * 100
# 6. Ewaluacja błędu ex-post prognozy wygasłej (na probie testowej)
ts_test = test_data['Pogoda'].values
rmse_test = np.sqrt(np.sum(np.square(ts_test - weather_forecast)) / len(ts_test))
mape_test = mape(ts_test, weather_forecast)
rmspe_test = (rmse_test / np.mean(ts_test)) * 100
print("--- Wyniki dopasowania ex-post Pogody (na okresie modelu) ---")
print(f"RMSE:  {rmse_val:.2f}")
print(f"MAPE:  {mape_val:.2f}%")
print(f"RMSPE: {rmspe_val:.2f}%\n")
print("--- Teoretyczne błędy ex-ante prognozy (Czerwiec-Sierpień 2018) ---")
for d, p, sp, vp in zip(test_data['okres'].dt.strftime('%Y-%m'), weather_forecast, s_p_ex_ante, v_p_ex_ante):
    print(f"{d} - Prognoza: {p:.2f} °C (Ex-ante: +/- {sp:.2f} °C, {vp:.2f}%)")
print("\n--- Błędy ex-post prognozy wygasłej (rzeczywiste odchylenia) ---")
for d, p, r in zip(test_data['okres'].dt.strftime('%Y-%m'), weather_forecast, ts_test):
    print(f"{d} - Prognoza: {p:.2f} °C, Rzeczywiste: {r:.2f} °C")
print(f"RMSE:  {rmse_test:.2f}")
print(f"MAPE:  {mape_test:.2f}%")
print(f"RMSPE: {rmspe_test:.2f}%")
# Generowanie wykresu
plt.figure(figsize=(10, 6))
mask_plot = train_data['okres'] >= '2017-01-01'
plt.plot(train_data[mask_plot]['okres'], ts_train[mask_plot], label='Rzeczywiste (Trening)', marker='o')
plt.plot(train_data[mask_plot]['okres'], fitted[mask_plot], label='Wartości teoretyczne (Trening)', marker='x', linestyle='--')
plt.plot(test_data['okres'], ts_test, label='Rzeczywiste (Test)', marker='o', color='green')
plt.fill_between(test_data['okres'], weather_forecast - s_p_ex_ante, weather_forecast + s_p_ex_ante, color='red', alpha=0.2, label='Błąd ex-ante (+/- 1 odch. std)')
plt.plot(test_data['okres'], weather_forecast, label='Prognoza (Ex-ante)', marker='X', linestyle='--', color='red')
plt.title("Prognoza temperatury - Pogoda (Holt-Winters)")
plt.xlabel("Data")
plt.ylabel("Temperatura (°C)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("prognoza_pogoda_hw_warunkowa.png")
print("\nWykres zapisano jako prognoza_pogoda_hw_warunkowa.png")
with open("pogoda_summary.txt", "w", encoding="utf-8") as f:
    f.write("--- Podsumowanie modelu Pogody (Holt-Winters) ---\n\n")
    f.write("1. Błędy dopasowania ex-post na okresie uczącym:\n")
    f.write(f"RMSE:  {rmse_val:.2f}\n")
    f.write(f"MAPE:  {mape_val:.2f}%\n")
    f.write(f"RMSPE: {rmspe_val:.2f}%\n\n")
    
    f.write("2. Teoretyczne błędy ex-ante prognozy (Czerwiec-Sierpień 2018):\n")
    for d, p, sp, vp in zip(test_data['okres'].dt.strftime('%Y-%m'), weather_forecast, s_p_ex_ante, v_p_ex_ante):
        f.write(f"{d} - Prognoza: {p:.2f} °C (Sp: +/- {sp:.2f} °C, Vp: {vp:.2f}%)\n")
        
    f.write("\n3. Błędy ex-post prognozy wygasłej (porównanie z rzeczywistością):\n")
    f.write(f"RMSE:  {rmse_test:.2f}\n")
    f.write(f"MAPE:  {mape_test:.2f}%\n")
    f.write(f"RMSPE: {rmspe_test:.2f}%\n")
