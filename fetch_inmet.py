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

def slugify(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text)
    return text.strip("_").lower()

TARGET_SLUG = slugify("SAO LUIZ DO PARAITINGA")

def get(session: requests.Session, url: str, **kw) -> requests.Response:
    r = session.get(url, timeout=60, **kw)
    r.raise_for_status()
    return r

def find_year_links(html: str) -> Dict[int, str]:
    soup = BeautifulSoup(html, "html.parser")
    links = {}
    for a in soup.find_all("a"):
        text = (a.get_text() or "").upper()
        m = re.search(r"ANO\s+(\d{4}).*AUTOM", text)
        href = a.get("href")
        if m and href:
            y = int(m.group(1))
            if not href.startswith("http"):
                href = "https://portal.inmet.gov.br" + href
            links[y] = href
    return links

def find_zip_links(html: str) -> List[str]:
    soup = BeautifulSoup(html, "html.parser")
    out = []
    for a in soup.find_all("a"):
        href = a.get("href") or ""
        if href.lower().endswith(".zip"):
            if not href.startswith("http"):
                href = "https://portal.inmet.gov.br" + href
            out.append(href)
    return out

def iter_csv_from_zip(content: bytes) -> Iterable[Tuple[str, bytes]]:
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        for info in zf.infolist():
            if info.filename.lower().endswith(".csv"):
                with zf.open(info) as f:
                    yield info.filename, f.read()

def try_read_csv(bytes_content: bytes) -> pd.DataFrame:
    encodings = ["latin-1", "utf-8-sig", "utf-8"]
    for enc in encodings:
        for sep in [";", ",", None]:
            try:
                return pd.read_csv(
                    io.BytesIO(bytes_content),
                    sep=sep,
                    encoding=enc,
                    skip_blank_lines=True,
                    engine="python",
                    on_bad_lines="skip"  # ignora linhas quebradas
                )
            except pd.errors.ParserError:
                continue
            except Exception:
                continue
    print("⚠ Falha na leitura: formato inesperado.")
    return pd.DataFrame()

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    def norm(c):
        c2 = unicodedata.normalize("NFKD", str(c)).encode("ascii","ignore").decode("ascii").upper()
        c2 = re.sub(r"[^A-Z0-9]+","_", c2).strip("_")
        return c2
    df = df.rename(columns={c: norm(c) for c in df.columns})
    if "DATA" in df.columns:
        df["DATA"] = pd.to_datetime(df["DATA"], errors="coerce", dayfirst=True)
    return df

def is_target_station(filename: str) -> bool:
    return TARGET_SLUG in slugify(filename)

def download_and_extract_for_year(session: requests.Session, year: int, out_raw: Path):
    print(f"Processando ano {year}...")
    res = get(session, BASE_URL)
    year_links = find_year_links(res.text)
    if year not in year_links:
        print(f"Nenhum link encontrado para o ano {year}.")
        return []

    year_url = year_links[year]
    res_y = get(session, year_url)

    content_type = res_y.headers.get("Content-Type", "").lower()
    if "text/html" not in content_type:
        print(f"[{year}] Link direto para arquivo detectado.")
        zip_links = [year_url]
    else:
        try:
            zip_links = find_zip_links(res_y.text)
        except Exception as e:
            print(f"[{year}] Erro ao parsear HTML: {e}")
            return []

    out_dfs = []
    for zurl in tqdm(zip_links, desc=f"{year} - zips"):
        try:
            r = get(session, zurl, stream=True)
            content = r.content
            out_raw.mkdir(parents=True, exist_ok=True)
            with open(out_raw / f"{year}_{os.path.basename(zurl)}", "wb") as f:
                f.write(content)
            for fname, bytes_csv in iter_csv_from_zip(content):
                if is_target_station(fname):
                    df = try_read_csv(bytes_csv)
                    if not df.empty:
                        out_dfs.append(normalize_columns(df))
        except Exception as e:
            print(f"[{year}] Falha no arquivo {zurl}: {e}")
    return out_dfs

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

    if args.years == "all":
        years = list(range(2000, datetime.now().year + 1))
    elif "-" in args.years:
        a, b = args.years.split("-")
        years = list(range(int(a), int(b) + 1))
    else:
        years = [int(x) for x in re.split(r"[,\s]+", args.years.strip()) if x]

    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})

    # CORREÇÃO: Inicializar all_dfs antes do loop
    all_dfs = []

    for y in years:
        dfs = download_and_extract_for_year(session, y, Path(args.raw_dir) / str(y))
        all_dfs.extend(dfs)

    if not all_dfs:
        print("Nenhum dado encontrado para nenhum ano.")
        return

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
