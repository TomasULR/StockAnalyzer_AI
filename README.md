# StockAnalyzer_AI

# AI Analýza Trhu S&P 500

Tento projekt je Python skript, který využívá umělou inteligenci a techniky strojového učení k analýze a predikci vývoje indexu S&P 500. Skript stahuje aktuální i historická data, počítá desítky technických indikátorů, trénuje několik prediktivních modelů a výsledky prezentuje formou přehledného reportu a interaktivního dashboardu.

## Klíčové Vlastnosti

-   **Automatizovaný sběr dat**: Stahuje nejnovější data S&P 500 pomocí `yfinance`.
-   **Pokročilá technická analýza**: Využívá knihovnu `TA-Lib` k výpočtu indikátorů jako RSI, MACD, Bollingerova pásma a další.
-   **Ensemble AI modelů**: Pro predikci využívá kombinaci několika modelů:
    -   Random Forest
    -   Gradient Boosting
    -   Vícevrstvá perceptronová síť (Neural Network)
    -   LSTM (Long Short-Term Memory) pro analýzu časových řad
-   **Analýza sentimentu**: Obsahuje simulovanou analýzu tržního sentimentu (lze rozšířit o reálná data z API).
-   **Interaktivní vizualizace**: Generuje přehledný a interaktivní dashboard pomocí `Plotly`.
-   **Detailní report**: Vypisuje do konzole souhrnný report s predikcemi, technickými signály a finálním doporučením.

## Použité Technologie

-   **Jazyk**: Python 3.8+
-   **Analýza dat**: Pandas, NumPy
-   **Strojové učení**: Scikit-learn, TensorFlow (Keras)
-   **Technické indikátory**: TA-Lib
-   **Sběr dat**: yfinance
-   **Vizualizace**: Plotly, Matplotlib, Seaborn
-   **Sentiment analýza**: NLTK, TextBlob

---

## Instalace a Nastavení

Pro spuštění skriptu je potřeba správně nastavit prostředí a nainstalovat všechny závislosti.

### 1. Klonování repozitáře

git clone https://your-repository-url.git
cd your-repository-directory

text

### 2. Vytvoření virtuálního prostředí (doporučeno)

python -m venv venv

Aktivace na Windows
venv\Scripts\activate

Aktivace na macOS/Linux
source venv/bin/activate

text

### 3. Instalace TA-Lib C knihovny

Toto je nejdůležitější krok, protože Python balíček `TA-Lib` vyžaduje, aby byla v systému nejprve nainstalována C knihovna.

-   **Windows**: Nejjednodušší je stáhnout předkompilovaný `.whl` soubor z [tohoto neoficiálního zdroje](https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib) (vyberte soubor odpovídající vaší verzi Pythonu a architektuře systému) a nainstalovat jej:
    ```
    pip install TA_Lib-0.4.28-cp311-cp311-win_amd64.whl
    ```

-   **macOS**: Použijte Homebrew:
    ```
    brew install ta-lib
    ```

-   **Linux (Debian/Ubuntu)**: Stáhněte a zkompilujte zdrojový kód, nebo použijte správce balíčků:
    ```
    # Stáhněte a rozbalte tar.gz z oficiálního webu
    ./configure --prefix=/usr
    make
    sudo make install
    ```

### 4. Instalace Python závislostí

Vytvořte soubor `requirements.txt` s následujícím obsahem:

yfinance
pandas
numpy
matplotlib
seaborn
scikit-learn
requests
textblob
nltk
TA-Lib
tensorflow
plotly

text

Poté nainstalujte všechny závislosti jedním příkazem:

pip install -r requirements.txt

text

### 5. Stažení dat pro NLTK

Pro sentiment analýzu je potřeba stáhnout VADER lexicon. Spusťte Python a zadejte:

import nltk
nltk.download('vader_lexicon')

text

---

## Použití

Po úspěšné instalaci všech závislostí spusťte hlavní skript z terminálu:

python sp500_ai_analyzer.py

text

Skript provede následující kroky:
1.  Stáhne historická data S&P 500.
2.  Vytvoří a připraví features pro modely.
3.  Natrénuje všechny AI modely (tento krok může chvíli trvat).
4.  Vypíše do konzole detailní **AI Analýza Report**.
5.  Otevře ve vašem prohlížeči **interaktivní dashboard** s vizualizacemi.

## Výstup

### 1. Konzolový Report

Report v konzoli obsahuje klíčové informace pro rychlé rozhodování:
-   Poslední zavírací cena.
-   Hodnoty klíčových indikátorů (RSI, MACD).
-   Predikce budoucího denního výnosu od každého AI modelu.
-   Průměrnou predikci ze všech modelů.
-   Finální doporučení (např. `BULLISH`, `BEARISH`, `NEUTRÁLNÍ`).

### 2. Interaktivní Dashboard

Dashboard vytvořený pomocí Plotly obsahuje několik grafů pro hlubší analýzu:
-   Historický vývoj ceny S&P 500 s klouzavými průměry.
-   Porovnání predikcí jednotlivých AI modelů.
-   Grafy technických indikátorů (RSI, MACD).
-   Analýza objemu obchodů.
-   Cena v kontextu Bollingerových pásem.
-   Důležitost jednotlivých features pro Random Forest model.

---

## Disclaimer

Tento nástroj je vytvořen **pouze pro vzdělávací a demonstrativní účely**. Predikce generované AI modely jsou založeny na historických datech a **nejsou finančním poradenstvím**. Minulá výkonnost není zárukou budoucích výsledků. Autor nenese žádnou odpovědnost za finanční ztráty vzniklé na základě informací z tohoto skriptu. Před jakýmkoliv investičním rozhodnutím se poraďte s kvalifikovaným finančním poradcem.
