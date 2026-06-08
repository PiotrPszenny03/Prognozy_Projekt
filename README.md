# Prognozy_Projekt

Projekt opiera się na bazie danych Prognozy_Baza_Danych.xlsx i składa się z trzech głównych skryptów wykonawczych:

1. run_model.py (Estymacja głównego modelu OLS) Skrypt bazowy służący do budowy ekonometrycznego modelu opisowego. Na podstawie próby uczącej (do maja 2018) estymuje wpływ zmiennych (pogoda, wakacje, ferie, wynagrodzenia) na zlogarytmowaną liczbę turystów. Plik wykonuje również testy diagnostyczne reszt (testy na normalność, brak autokorelacji, homoskedastyczność oraz prawidłową specyfikację) weryfikujące statystyczną poprawność modelu.

2. forecast_weather.py (Prognoza pomocnicza Holt-Wintersa) Skrypt odpowiedzialny za przygotowanie prognoz wartości zmiennych objaśniających (niezbędnych do wykonania prognozy warunkowej). Z wykorzystaniem modelu wygładzania wykładniczego Holt-Wintersa generuje prognozy out-of-sample dla zmiennej Pogoda na miesiące czerwiec-sierpień 2018. 

3. conditional_forecast.py (Finałowa prognoza warunkowa) Główny skrypt agregujący wyniki i generujący docelową predykcję na 3 letnie miesiące (czerwiec-sierpień 2018). Łączy główny model OLS z zaprognozowanymi wartościami pogody (z modelu Holt-Wintersa) oraz wprowadzonymi prognozami wynagrodzeń. Poza predykcją, skrypt liczy teoretyczny margines błędu ex-ante oraz konfrontuje uzyskane wyniki z rzeczywiście zaobserwowanymi danymi w celu wyliczenia empirycznych błędów wygasłych (ex-post), takich jak RMSE czy MAPE.
