import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
df = pd.read_excel("Prognozy_Baza_Danych.xlsx", sheet_name=0)
df['okres'] = pd.to_datetime(df['Zmienna'].str.replace(' M', '-'), format='%Y-%m')
df['okres'] = df['okres'] + pd.offsets.MonthEnd(0)
df['miesiąc'] = df['okres'].dt.month
# clean columns if needed
df['Wakacje'] = df['miesiąc'].isin([7, 8]).astype(int)
df['Ferie'] = df['miesiąc'].isin([1, 2]).astype(int)
# Filter from Feb 2015 to May 2018
df = df[(df['okres'] >= '2015-02-01') & (df['okres'] <= '2018-05-31')]
df = df.dropna(subset=['Wynagrodzenia'])
y = np.log(df['Turyści'])
X = pd.DataFrame()
X['Pogoda'] = df['Pogoda']  # Nie logarytmujemy, bo ma wartości ujemne
X['Wakacje'] = df['Wakacje']
X['Ferie'] = df['Ferie']
X['Wynagrodzenia'] = df['Wynagrodzenia']
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
# Plotting
plot_dates = df['okres'].dt.to_period('M').dt.to_timestamp()
plt.figure(figsize=(12, 6))
plt.plot(plot_dates, np.exp(y), label="Wartości rzeczywiste", marker='o', markersize=4)
plt.plot(plot_dates, np.exp(model.fittedvalues), label="Wartości teoretyczne", linestyle='--', marker='x', markersize=4)
plt.ticklabel_format(style='plain', axis='y')
plt.title("Liczba turystów - Wartości Rzeczywiste vs Teoretyczne")
plt.xlabel("Data")
plt.ylabel("Liczba turystów")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("wykres_teoretyczne_rzeczywiste.png")
import statsmodels.stats.api as sms
import statsmodels.stats.diagnostic as smd
from scipy import stats
# 1. Normality (Jarque-Bera from statsmodels, Shapiro-Wilk from scipy)
jb, jbpv, skew, kurtosis = sms.jarque_bera(model.resid)
shapiro_stat, shapiro_p = stats.shapiro(model.resid)
# 2. Heteroskedasticity (Breusch-Pagan)
bp_stat, bp_pval, _, _ = sms.het_breuschpagan(model.resid, model.model.exog)
# 3. Autocorrelation (Breusch-Godfrey)
bg_stat, bg_pval, _, _ = smd.acorr_breusch_godfrey(model, nlags=1)
# 4. RAMSEY RESET
reset_res = smd.linear_reset(model, power=2, test_type="fitted", use_f=True)
reset_stat = reset_res.fvalue
reset_pval = reset_res.pvalue
diag_text = f"""
--- DIAGNOSTYKA MODELU ---
1. Test normalności reszt:
   - Jarque-Bera: stat={jb:.4f}, p-value={jbpv:.4f}
   - Shapiro-Wilk: stat={shapiro_stat:.4f}, p-value={shapiro_p:.4f}
   (H0: Reszty mają rozkład normalny)
2. Test heteroskedastyczności (Breusch-Pagan):
   - stat={bp_stat:.4f}, p-value={bp_pval:.4f}
   (H0: Wariancja reszt jest stała / homoskedastyczność)
3. Test autokorelacji reszt (Breusch-Godfrey, lag=1):
   - LM-stat={bg_stat:.4f}, p-value={bg_pval:.4f}
   (H0: Brak autokorelacji)
4. Test specyfikacji RAMSEY RESET:
   - F-stat={float(reset_stat):.4f}, p-value={float(reset_pval):.4f}
   (H0: Prawidłowa specyfikacja modelu)
"""
with open("model_summary.txt", "w", encoding="utf-8") as f:
    f.write(model.summary().as_text())
    f.write("\n" + diag_text)