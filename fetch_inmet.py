import os, re, io, zipfile, argparse, unicodedata
from datetime import datetime
from pathlib import Path
from typing import List, Iterable, Tuple, Dict
import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_values
import requests
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()  # Carrega variáveis de ambiente

BASE_URL = "https://portal.inmet.gov.br/dadoshistoricos"

# ... (código existente até a função normalize_columns) ...

def get_neon_connection():
    """Cria e retorna uma conexão com o banco NEON."""
    try:
        conn = psycopg2.connect(os.getenv("NEON_DATABASE_URL"))
        return conn
    except Exception as e:
        print(f"Erro ao conectar ao NEON: {e}")
        raise

def upload_to_neon(df: pd.DataFrame):
    """Envia um DataFrame para o banco NEON, atualizando registros existentes."""
    if df.empty:
        return

    # Garante as colunas necessárias
    for col in ["DATA","TEMPERATURA_MEDIA","TEMPERATURA_MAXIMA","TEMPERATURA_MINIMA",
                "UMIDADE_RELATIVA","PRECIPITACAO","VELOCIDADE_VENTO","PRESSAO_ATMOSFERICA"]:
        if col not in df.columns:
            df[col] = None

    # Filtra e renomeia colunas
    df = df[["DATA","TEMPERATURA_MEDIA","TEMPERATURA_MAXIMA","TEMPERATURA_MINIMA",
             "UMIDADE_RELATIVA","PRECIPITACAO","VELOCIDADE_VENTO","PRESSAO_ATMOSFERICA"]].copy()
    df.rename(columns={
        "DATA": "data",
        "TEMPERATURA_MEDIA": "temperatura_media",
        "TEMPERATURA_MAXIMA": "temperatura_maxima",
        "TEMPERATURA_MINIMA": "temperatura_minima",
        "UMIDADE_RELATIVA": "umidade_relativa",
        "PRECIPITACAO": "precipitacao",
        "VELOCIDADE_VENTO": "velocidade_vento",
        "PRESSAO_ATMOSFERICA": "pressao_atmosferica"
    }, inplace=True)

    # Converte para lista de tuplas
    records = df.to_records(index=False).tolist()

    conn = None
    try:
        conn = get_neon_connection()
        cur = conn.cursor()

        # Query de inserção/atualização (UPSERT)
        query = sql.SQL("""
            INSERT INTO inmet_data (data, temperatura_media, temperatura_maxima, temperatura_minima,
                                   umidade_relativa, precipitacao, velocidade_vento, pressao_atmosferica)
            VALUES %s
            ON CONFLICT (data) DO UPDATE
            SET temperatura_media = EXCLUDED.temperatura_media,
                temperatura_maxima = EXCLUDED.temperatura_maxima,
                temperatura_minima = EXCLUDED.temperatura_minima,
                umidade_relativa = EXCLUDED.umidade_relativa,
                precipitacao = EXCLUDED.precipitacao,
                velocidade_vento = EXCLUDED.velocidade_vento,
                pressao_atmosferica = EXCLUDED.pressao_atmosferica;
        """)

        execute_values(cur, query, records)
        conn.commit()
        print(f"✔ {len(records)} registros inseridos/atualizados no NEON.")
    except Exception as e:
        print(f"❌ Erro ao enviar para o NEON: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()

def main():
    parser = argparse.ArgumentParser(description="Baixa dados do INMET e envia para o NEON.")
    parser.add_argument("--years", default="all")
    parser.add_argument("--raw_dir", default="data/raw")
    args = parser.parse_args()

    # ... (código existente para coleta de dados) ...

    # Filtra apenas DataFrames com a coluna DATA
    valid_dfs = [df for df in all_dfs if "DATA" in df.columns and not df.empty]
    if not valid_dfs:
        print("Nenhum dado válido com coluna DATA encontrado.")
        return

    df_all = pd.concat(valid_dfs, ignore_index=True)
    df_all = df_all.dropna(subset=["DATA"]).sort_values("DATA").drop_duplicates(subset=["DATA"])

    # Envia para o NEON em vez de salvar CSV
    upload_to_neon(df_all)

if __name__ == "__main__":
    main()