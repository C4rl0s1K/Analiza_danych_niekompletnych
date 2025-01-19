# 1. Wstęp

W projekcie zajmowałam się budową modelu predykcyjnego, który przewiduje ceny wynajmu mieszkań, korzystając z danych rynkowych. Wykorzystałam różne modele regresyjne, a ostatecznym wyborem okazał się XGBoost, który osiągnął wynik około 95933.45 MSE na platformie Kaggle (mój nick: Anita Carlos). Analiza danych objęła identyfikację braków i ich imputację oraz kodowanie zmiennych kategorycznych. W trakcie pracy LLM okazały się pomocne przy rozwiązywaniu problemów kodowych i wyszukiwaniu optymalnych technik przetwarzania danych.

# 2. Metodyka

Na początku projektu przeprowadziłam eksploracyjną analizę danych (EDA), aby lepiej zrozumieć strukturę dostępnych informacji i zidentyfikować potencjalne problemy. Sprawdziłam, czy w zbiorze danych występują braki, a tam, gdzie je znalazłam, zastosowałam odpowiednie techniki imputacji, aby uzupełnić brakujące wartości.

Kluczowym krokiem było przygotowanie danych, które składały się zarówno ze zmiennych numerycznych, jak i kategorycznych. W pierwszej kolejności zaimplementowałam imputację brakujących danych. Dla zmiennych numerycznych zastosowałam metodę uzupełniania medianą, co pozwoliło zminimalizować wpływ ekstremalnych wartości na model. Z kolei dla zmiennych kategorycznych użyłam strategii najczęstszej wartości ("most_frequent"), aby zachować spójność kategorii bez wprowadzania szumu informacyjnego. Do uzupenienia brakujcych danych w zbiorach danych zastosowano SimpleImputer z biblioteki Scikit-learn.

Kolejnym ważnym etapem było przekształcenie kolumn związanych z datami na zmienne numeryczne, reprezentujące liczbę dni aktywności ogłoszenia oraz pozostały czas do jego wygaśnięcia. Takie podejście pozwoliło mi uchwycić dynamikę czasową, co znacząco poprawiło jakość predykcji. Dla zmiennych kategorycznych wybrałam różne techniki kodowania. W przypadku zmiennej "quarter" zastosowałam kodowanie One-Hot Encoding, które zapobiega problemom wynikającym z przypisywania dzielnic do danego mieszkania. Pozostałe zmienne binarne zakodowałam za pomocą Label Encoding, co było odpowiednie ze względu na ich logiczny, dwuwartościowy charakter.

Dodatkowo sprawdziłam zgodność kolumn między zbiorem treningowym a testowym, dodając brakujące cechy jako kolumny z wartością zerową tam, gdzie było to konieczne. Dzięki temu dane były przygotowane w sposób zapewniający zgodność strukturalną, co umożliwiło efektywne działanie modelu. Ostatecznie, podzieliłam dane na zestaw treningowy i testowy w proporcji 80:20, co pozwoliło mi ocenić rzeczywistą skuteczność modelu na nowych danych.

# 3. Wyniki

Podczas eksperymentów przetestowałam trzy różne modele regresyjne. Początkowo zastosowałam model LinearRegression(), który uzyskał wynik MSE na poziomie powyżej 3 milionów, co wskazywało na jego niską przydatność dla tego zadania. Następnie skorzystałam z modelu RandomForestRegressor z 100 estymatorami, co zmniejszyło MSE do około 111 000. Zwiększenie liczby estymatorów do 1000 poprawiło wynik do około 97 000. MSE było wyliczane na zbiorze treningowym, natomiast sprawdzając swoje wyniki na kagglu, wartość MSE dla 1000 estyamtorów wynosiła 100 393.15.

Ostatecznie zastosowałam model XGBoost, co pozwoliło uzyskać najlepszy wynik MSE na poziomie poniżej 87738.37 na zbiorze treningowym. Model ten był testowany kilkukrotnie z różnymi parametrami, co pozwoliło zoptymalizować jego wydajność. Wynik ten został przesłany na platformę Kaggle, gdzie zajął wyższą pozycję niż poprzednie modele, potwierdzając efektywność podejścia i lepszy dobór parametrów.

# 4. Podsumowanie

W trakcie realizacji projektu kluczowe dla mnie było zrozumienie wpływu różnych technik przygotowania danych i doboru modeli na wynik predykcji. Proces imputacji danych pozwolił na zachowanie pełnego zbioru danych, jednak jej wpływ na poprawę wyników był niewielki w moim odczuciu. Znacznie większy wpływ miało znalezienie odpowiedniego modulu, czy później odpowiednie dostrojenie parametrów modelu XGBoost, który pozwolił osiągnąć najniższy wynik MSE

Wybór odpowiedniego modelu opierał się na badaniach dotyczących metod regresji dla dużych zbiorów danych, co zwiększyło moją świadomość w zakresie modelowania danych. Podczas tego procesu nie korzystałam z dużych modeli językowych bezpośrednio do tworzenia predykcji, ale LLM okazały się pomocne w rozwiązywaniu problemów programistycznych oraz w eksploracji technik uczenia maszynowego.

# 5. Kody do odtworzenia wyników

```python
# Importowanie odpowiednich bibliotek
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


# Utworzenie zbioru treningowego oraz testowego

train_data = pd.read_csv("raw data/pzn-rent-train.csv")
test_data = pd.read_csv("raw data/pzn-rent-test.csv")


# Podzielenie zbioru danych na dane kategoryczne (True/False) i numeryczne

numerical_columns = train_data.select_dtypes(include=["float64", "int64"]).columns.tolist()
categorical_columns = train_data.select_dtypes(include=["object"]).columns.tolist()

numerical_columns.remove("price")

# Tworze imputer, który uzupełnij odpowiednio dane numeryczne

numerical_imputer = SimpleImputer(strategy="median")
train_data[numerical_columns] = numerical_imputer.fit_transform(train_data[numerical_columns])
test_data[numerical_columns] = numerical_imputer.transform(test_data[numerical_columns])

# Tworze imputer, który uzupełnij odpowiednio dane kategoryczne

categorical_imputer = SimpleImputer(strategy="most_frequent")
train_data[categorical_columns] = categorical_imputer.fit_transform(train_data[categorical_columns])
test_data[categorical_columns] = categorical_imputer.transform(test_data[categorical_columns])

# Przekształcenie kolumn z typu numerycznego na typ daty

train_data["date_activ"] = pd.to_datetime(train_data["date_activ"])
train_data["date_modif"] = pd.to_datetime(train_data["date_modif"])
train_data["date_expire"] = pd.to_datetime(train_data["date_expire"])

test_data["date_activ"] = pd.to_datetime(test_data["date_activ"])
test_data["date_modif"] = pd.to_datetime(test_data["date_modif"])
test_data["date_expire"] = pd.to_datetime(test_data["date_expire"])

# Pozbycie się dat, na konkretna liczbe dni aby model mógł działać poprawnie

train_data["active_duration"] = (train_data["date_modif"] - train_data["date_activ"]).dt.days
train_data["remaining_days"] = (train_data["date_expire"] - train_data["date_modif"]).dt.days

test_data["active_duration"] = (test_data["date_modif"] - test_data["date_activ"]).dt.days
test_data["remaining_days"] = (test_data["date_expire"] - test_data["date_modif"]).dt.days


train_data.drop(["date_activ", "date_modif", "date_expire"], axis=1, inplace=True)
test_data.drop(["date_activ", "date_modif", "date_expire"], axis=1, inplace=True)

# Lista kolumn kategorycznych do przetworzenia

categorical_columns = ["individual", "flat_furnished", "flat_for_students", "flat_balcony",
                       "flat_garage", "flat_basement", "flat_garden", "flat_tarrace", "flat_lift",
                       "flat_two_level", "flat_kitchen_sep", "flat_air_cond", "flat_nonsmokers",
                       "flat_washmachine", "flat_dishwasher", "flat_internet", "flat_television",
                       "flat_anti_blinds", "flat_monitoring", "flat_closed_area", "quarter"]

# Zastosowanie kodowania One-Hot Encoding dla zmiennej "quarter"
train_data = pd.get_dummies(train_data, columns=["quarter"], drop_first=True)
test_data = pd.get_dummies(test_data, columns=["quarter"], drop_first=True)

encoder = LabelEncoder()

# Iteracja przez kolumny kategoryczne i kodowanie za pomocą LabelEncoder
for col in categorical_columns:
    if col != "quarter": 
        train_data[col] = encoder.fit_transform(train_data[col])
        test_data[col] = encoder.transform(test_data[col])

# Sprawdzam kolumny w zbiorach

train_columns = train_data.columns
test_columns = test_data.columns


# Dodaj brakujące kolumny do zbioru testowego

for column in train_columns:
    if column not in test_columns:
        test_data[column] = 0

# Dodaj brakujące kolumny do zbioru treningowego

for column in test_columns:
    if column not in train_columns:
        train_data[column] = 0


# Przygotowanie danych do predykcji
X = train_data.drop(columns=["id", "ad_title", "price"])  # Usunięcie kolumn niepotrzebnych
y = train_data["price"] #Cena jest zmienna która będzie predykcjonowana 

# Podział danych na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_xgb = xgb.XGBRegressor(n_estimators=3000, learning_rate=0.005, max_depth=7, min_child_weight=5, subsample=0.7, colsample_bytree=0.7, gamma=0.1, random_state=42)

model_xgb.fit(X_train, y_train)

# Predykcja na zbiorze testowym
y_pred_xgb = model_xgb.predict(X_test)

# Obliczanie MSE
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
print(f"Mean Squared Error XGBoost Regressor: {mse_xgb}")

# Przygotowanie zbioru testowego
X_test_final = test_data.drop(columns=["id", "ad_title"])  # Usunięcie kolumn niepotrzebnych
X_test_final = X_test_final[X_train.columns]  # Upewnienie się, że kolumny są w tej samej kolejności co w X_train

# Predykcja na zbiorze testowym
y_test_predictions = model_xgb.predict(X_test_final)

submission = pd.DataFrame({
    "ID": test_data.index + 1,
    "TARGET": y_test_predictions
})

print(submission.head())
submission.to_csv("pzn-solution2.csv", index=False)



