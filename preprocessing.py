import pandas as pd

def preprocess_data(df):
    """Função de pré-processamento para ML"""
    df = df.copy()
    
    # Preencher valores faltantes
    for col in ["TEMPERATURA_MEDIA", "UMIDADE_RELATIVA"]:
        df[col] = df[col].fillna(df[col].mean())
    
    # Engenharia de features
    df["HORA"] = df["DATA"].dt.hour
    df["DIA_DA_SEMANA"] = df["DATA"].dt.dayofweek
    
    # Transformações
    df["PRECIPITACAO_LOG"] = np.log1p(df["PRECIPITACAO"])
    
    return df
