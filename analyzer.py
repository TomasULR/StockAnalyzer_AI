class SP500AIAnalyzer:
    def __init__(self, period="5y"):
        """
        Inicializace analyzátoru S&P 500 s AI funkcionalitou
        """
        self.period = period
        self.sp500_data = None
        self.features = None
        self.target = None
        self.models = {}
        self.predictions = {}
        self.scaler = StandardScaler()
        
    def fetch_sp500_data(self):
        """
        Stažení historických dat S&P 500
        """
        print("Stahuji data S&P 500...")
        ticker = "^GSPC"  # S&P 500 ticker
        self.sp500_data = yf.download(ticker, period=self.period)
        
        # Přidání dodatečných ukazatelů
        self.sp500_data['Returns'] = self.sp500_data['Adj Close'].pct_change()
        self.sp500_data['Log_Returns'] = np.log(self.sp500_data['Adj Close'] / self.sp500_data['Adj Close'].shift(1))
        
        # Volatilita (rolling standard deviation)
        self.sp500_data['Volatility'] = self.sp500_data['Returns'].rolling(window=20).std()
        
        # Moving averages
        self.sp500_data['MA_20'] = self.sp500_data['Adj Close'].rolling(window=20).mean()
        self.sp500_data['MA_50'] = self.sp500_data['Adj Close'].rolling(window=50).mean()
        self.sp500_data['MA_200'] = self.sp500_data['Adj Close'].rolling(window=200).mean()
        
        # RSI
        self.sp500_data['RSI'] = talib.RSI(self.sp500_data['Adj Close'].values, timeperiod=14)
        
        # MACD
        macd, macd_signal, macd_hist = talib.MACD(self.sp500_data['Adj Close'].values)
        self.sp500_data['MACD'] = macd
        self.sp500_data['MACD_Signal'] = macd_signal
        self.sp500_data['MACD_Hist'] = macd_hist
        
        # Bollinger Bands
        upper, middle, lower = talib.BBANDS(self.sp500_data['Adj Close'].values)
        self.sp500_data['BB_Upper'] = upper
        self.sp500_data['BB_Middle'] = middle
        self.sp500_data['BB_Lower'] = lower
        
        # Volume indicators
        self.sp500_data['Volume_MA'] = self.sp500_data['Volume'].rolling(window=20).mean()
        self.sp500_data['Volume_Ratio'] = self.sp500_data['Volume'] / self.sp500_data['Volume_MA']
        
        print(f"Data stažena: {len(self.sp500_data)} záznamů")
        return self.sp500_data

    def create_advanced_features(self):
        """
        Vytvoření pokročilých features pro AI model
        """
        df = self.sp500_data.copy()
        
        # Technické indikátory
        df['Price_Change'] = df['Adj Close'].pct_change()
        df['High_Low_Ratio'] = df['High'] / df['Low']
        df['Volume_Price_Trend'] = df['Volume'] * df['Price_Change']
        
        # Momentum indicators
        df['ROC_5'] = df['Adj Close'].pct_change(5)  # 5-day Rate of Change
        df['ROC_10'] = df['Adj Close'].pct_change(10)  # 10-day Rate of Change
        
        # Volatility clustering
        df['Volatility_MA'] = df['Volatility'].rolling(window=10).mean()
        df['Volatility_Ratio'] = df['Volatility'] / df['Volatility_MA']
        
        # Price position relative to moving averages
        df['Price_vs_MA20'] = df['Adj Close'] / df['MA_20']
        df['Price_vs_MA50'] = df['Adj Close'] / df['MA_50']
        df['Price_vs_MA200'] = df['Adj Close'] / df['MA_200']
        
        # Bollinger Band position
        df['BB_Position'] = (df['Adj Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # Lag features (pro časové řady)
        for lag in [1, 2, 3, 5]:
            df[f'Returns_Lag_{lag}'] = df['Returns'].shift(lag)
            df[f'Volume_Lag_{lag}'] = df['Volume'].shift(lag)
            df[f'RSI_Lag_{lag}'] = df['RSI'].shift(lag)
        
        # Vytvoření target variable (budoucí return)
        df['Future_Return'] = df['Returns'].shift(-1)  # Predikce příštího dne
        
        # Vyčištění dat
        df = df.dropna()
        
        # Výběr features pro model
        feature_columns = [
            'Returns', 'Volatility', 'RSI', 'MACD', 'MACD_Hist',
            'BB_Position', 'Volume_Ratio', 'Price_vs_MA20', 'Price_vs_MA50',
            'ROC_5', 'ROC_10', 'Volatility_Ratio', 'High_Low_Ratio',
            'Returns_Lag_1', 'Returns_Lag_2', 'Returns_Lag_3',
            'Volume_Lag_1', 'RSI_Lag_1', 'Volume_Price_Trend'
        ]
        
        self.features = df[feature_columns]
        self.target = df['Future_Return']
        
        print(f"Vytvořeno {len(feature_columns)} features pro AI model")
        return self.features, self.target

    def train_ai_models(self):
        """
        Trénování různých AI modelů pro predikci
        """
        X_train, X_test, y_train, y_test = train_test_split(
            self.features, self.target, test_size=0.2, random_state=42, shuffle=False
        )
        
        # Standardizace dat
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 1. Random Forest Model
        print("Trénuji Random Forest model...")
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train_scaled, y_train)
        self.models['Random_Forest'] = rf_model
        
        # 2. Gradient Boosting Model
        print("Trénuji Gradient Boosting model...")
        gb_model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        gb_model.fit(X_train_scaled, y_train)
        self.models['Gradient_Boosting'] = gb_model
        
        # 3. Neural Network Model
        print("Trénuji Neural Network model...")
        nn_model = MLPRegressor(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            alpha=0.001,
            max_iter=1000,
            random_state=42
        )
        nn_model.fit(X_train_scaled, y_train)
        self.models['Neural_Network'] = nn_model
        
        # 4. LSTM Model pro časové řady
        print("Trénuji LSTM model...")
        lstm_model = self.create_lstm_model(X_train_scaled.shape[1])
        
        # Reshape data pro LSTM
        X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
        X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))
        
        lstm_model.fit(X_train_lstm, y_train, epochs=50, batch_size=32, verbose=0)
        self.models['LSTM'] = lstm_model
        
        # Evaluace modelů
        self.evaluate_models(X_test_scaled, y_test, X_test_lstm)
        
        print("Všechny AI modely úspěšně natrénovány!")
        
    def create_lstm_model(self, input_shape):
        """
        Vytvoření LSTM modelu pro časové řady
        """
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(1, input_shape)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model
        
    def evaluate_models(self, X_test_scaled, y_test, X_test_lstm):
        """
        Evaluace výkonnosti všech modelů
        """
        print("\n=== EVALUACE AI MODELŮ ===")
        
        for name, model in self.models.items():
            if name == 'LSTM':
                predictions = model.predict(X_test_lstm).flatten()
            else:
                predictions = model.predict(X_test_scaled)
            
            mse = mean_squared_error(y_test, predictions)
            mae = mean_absolute_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            
            self.predictions[name] = predictions
            
            print(f"\n{name}:")
            print(f"  MSE: {mse:.6f}")
            print(f"  MAE: {mae:.6f}")
            print(f"  R²: {r2:.4f}")

    def get_market_sentiment(self):
        """
        Analýza tržního sentimentu z finančních zpráv
        """
        try:
            # Simulace sentiment analýzy (v reálné aplikaci by se použily API jako NewsAPI)
            # Zde vytvoříme ukázkový sentiment score
            
            # V reálné implementaci bychom použili:
            # - NewsAPI pro získání zpráv
            # - VADER sentiment analyzer pro analýzu
            # - GPT modely pro pokročilou analýzu
            
            sentiment_scores = {
                'positive': np.random.uniform(0.3, 0.7),
                'negative': np.random.uniform(0.1, 0.4),
                'neutral': np.random.uniform(0.2, 0.5)
            }
            
            # Normalizace skóre
            total = sum(sentiment_scores.values())
            sentiment_scores = {k: v/total for k, v in sentiment_scores.items()}
            
            # Overall sentiment score (-1 až 1)
            overall_sentiment = sentiment_scores['positive'] - sentiment_scores['negative']
            
            print(f"\n=== TRŽNÍ SENTIMENT ===")
            print(f"Pozitivní: {sentiment_scores['positive']:.3f}")
            print(f"Negativní: {sentiment_scores['negative']:.3f}")
            print(f"Neutrální: {sentiment_scores['neutral']:.3f}")
            print(f"Celkový sentiment: {overall_sentiment:.3f}")
            
            return overall_sentiment, sentiment_scores
            
        except Exception as e:
            print(f"Chyba při analýze sentimentu: {e}")
            return 0, {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34}

    def create_comprehensive_analysis(self):
        """
        Vytvoření komprehensivní analýzy s vizualizacemi
        """
        # Získání sentiment dat
        sentiment_score, sentiment_breakdown = self.get_market_sentiment()
        
        # Vytvoření subplotů
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'S&P 500 Cena s Moving Averages',
                'AI Model Predictions Comparison',
                'Technické Indikátory (RSI, MACD)',
                'Volume Analysis',
                'Volatilita a Bollinger Bands',
                'Feature Importance (Random Forest)'
            ),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": True}, {"secondary_y": False}],
                   [{"secondary_y": True}, {"secondary_y": False}]]
        )
        
        # Plot 1: Cena s MA
        recent_data = self.sp500_data.tail(252)  # Posledních 252 dní
        fig.add_trace(
            go.Scatter(x=recent_data.index, y=recent_data['Adj Close'], 
                      name='S&P 500', line=dict(color='blue')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=recent_data.index, y=recent_data['MA_20'], 
                      name='MA 20', line=dict(color='orange')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=recent_data.index, y=recent_data['MA_50'], 
                      name='MA 50', line=dict(color='red')),
            row=1, col=1
        )
        
        # Plot 2: AI Predictions Comparison
        if self.predictions:
            test_dates = recent_data.index[-len(list(self.predictions.values())[0]):]
            for model_name, predictions in self.predictions.items():
                fig.add_trace(
                    go.Scatter(x=test_dates, y=predictions, 
                              name=f'{model_name} Prediction', mode='lines'),
                    row=1, col=2
                )
        
        # Plot 3: RSI
        fig.add_trace(
            go.Scatter(x=recent_data.index, y=recent_data['RSI'], 
                      name='RSI', line=dict(color='purple')),
            row=2, col=1
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        # Plot 4: Volume
        fig.add_trace(
            go.Bar(x=recent_data.index, y=recent_data['Volume'], 
                   name='Volume', marker_color='lightblue'),
            row=2, col=2
        )
        
        # Plot 5: Bollinger Bands
        fig.add_trace(
            go.Scatter(x=recent_data.index, y=recent_data['BB_Upper'], 
                      name='BB Upper', line=dict(color='gray')),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(x=recent_data.index, y=recent_data['Adj Close'], 
                      name='Price', line=dict(color='blue')),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(x=recent_data.index, y=recent_data['BB_Lower'], 
                      name='BB Lower', line=dict(color='gray')),
            row=3, col=1
        )
        
        # Plot 6: Feature Importance
        if 'Random_Forest' in self.models:
            importances = self.models['Random_Forest'].feature_importances_
            feature_names = self.features.columns
            sorted_indices = np.argsort(importances)[::-1][:10]
            
            fig.add_trace(
                go.Bar(x=feature_names[sorted_indices], y=importances[sorted_indices],
                       name='Feature Importance'),
                row=3, col=2
            )
        
        fig.update_layout(height=1200, title_text="S&P 500 AI Analýza Dashboard")
        fig.show()
        
        # Vytvoření predikčního reportu
        self.generate_prediction_report(sentiment_score)
        
    def generate_prediction_report(self, sentiment_score):
        """
        Generování detailního predikčního reportu
        """
        print("\n" + "="*60)
        print("           S&P 500 AI ANALÝZA REPORT")
        print("="*60)
        
        # Získání nejnovějších dat
        latest_data = self.sp500_data.iloc[-1]
        latest_price = latest_data['Adj Close']
        
        print(f"\nAktuální data:")
        print(f"  Poslední cena: ${latest_price:.2f}")
        print(f"  RSI: {latest_data['RSI']:.2f}")
        print(f"  MACD: {latest_data['MACD']:.4f}")
        print(f"  Volatilita: {latest_data['Volatility']:.4f}")
        
        # AI Predikce
        if self.predictions:
            print(f"\nAI Predikce (příští return):")
            avg_prediction = np.mean(list(self.predictions.values()))
            for model_name, predictions in self.predictions.items():
                latest_pred = predictions[-1]
                print(f"  {model_name}: {latest_pred:.4f}")
            
            print(f"  Průměrná predikce: {avg_prediction:.4f}")
            
            # Interpretace
            if avg_prediction > 0.005:
                recommendation = "BULLISH - Silný nárůst očekáván"
            elif avg_prediction > 0.001:
                recommendation = "MÍRNĚ BULLISH - Mírný nárůst očekáván"
            elif avg_prediction > -0.001:
                recommendation = "NEUTRÁLNÍ - Sideways pohyb"
            elif avg_prediction > -0.005:
                recommendation = "MÍRNĚ BEARISH - Mírný pokles očekáván"
            else:
                recommendation = "BEARISH - Silný pokles očekáván"
                
            print(f"\n  DOPORUČENÍ: {recommendation}")
        
        # Technická analýza
        print(f"\nTechnická analýza:")
        rsi = latest_data['RSI']
        if rsi > 70:
            rsi_signal = "PŘEKOUPENO"
        elif rsi < 30:
            rsi_signal = "PŘEPRODÁNO"
        else:
            rsi_signal = "NEUTRÁLNÍ"
        print(f"  RSI signál: {rsi_signal}")
        
        # MA signály
        price_vs_ma20 = latest_data['Adj Close'] / latest_data['MA_20']
        if price_vs_ma20 > 1.02:
            ma_signal = "BULLISH - Cena výrazně nad MA20"
        elif price_vs_ma20 > 1:
            ma_signal = "MÍRNĚ BULLISH - Cena nad MA20"
        elif price_vs_ma20 > 0.98:
            ma_signal = "NEUTRÁLNÍ"
        else:
            ma_signal = "BEARISH - Cena pod MA20"
        print(f"  MA signál: {ma_signal}")
        
        # Sentiment
        print(f"\nTržní sentiment: {sentiment_score:.3f}")
        if sentiment_score > 0.2:
            sentiment_interpretation = "POZITIVNÍ"
        elif sentiment_score > -0.2:
            sentiment_interpretation = "NEUTRÁLNÍ"
        else:
            sentiment_interpretation = "NEGATIVNÍ"
        print(f"  Interpretace: {sentiment_interpretation}")
        
        print("\n" + "="*60)
