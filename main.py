def main():
    """
    Hlavní funkce pro spuštění AI analýzy S&P 500
    """
    # Inicializace analyzátoru
    analyzer = SP500AIAnalyzer(period="2y")
    
    # Stažení dat
    analyzer.fetch_sp500_data()
    
    # Vytvoření features
    analyzer.create_advanced_features()
    
    # Trénování AI modelů
    analyzer.train_ai_models()
    
    # Vytvoření komprehensivní analýzy
    analyzer.create_comprehensive_analysis()
    
    print("\nAI analýza S&P 500 dokončena!")
    
    return analyzer

if __name__ == "__main__":
    # Spuštění analýzy
    sp500_analyzer = main()
    
    # Možnost pro periodické aktualizace
    # schedule.every(1).hours.do(lambda: sp500_analyzer.update_analysis())
