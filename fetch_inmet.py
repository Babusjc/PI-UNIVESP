import os, re, io, zipfile, argparse, unicodedata, time
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
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36"
]

# Código da estação alvo
STATION_CODE = "A740"

def normalize_station_name(name: str) -> str:
    """Normaliza o nome da estação para facilitar a comparação."""
    name = name.lower()
    # Substitui variações de escrita
    name = re.sub(r"[^a-z0-9]+", " ", name)  # Remove caracteres especiais
    name = re.sub(r"\bsao\b", "são", name)    # Padroniza "sao" para "são"
    name = re.sub(r"\bluis\b", "luiz", name)  # Padroniza "luis" para "luiz"
    return name.strip()

def is_target_station(filename: str) -> bool:
    """Verifica se o arquivo é da estação desejada pelo código ou nome normalizado."""
    normalized = normalize_station_name(filename)
    # Verifica pelo código da estação ou pelo nome padronizado
    return STATION_CODE.lower() in normalized or "são luiz do paraitinga" in normalized

def get(session: requests.Session, url: str, retries=3, delay=5, **kw) -> requests.Response:
    for attempt in range(retries):
        try:
            r = session.get(url, timeout=120, **kw)
            r.raise_for_status()
            return r
        except requests.exceptions.RequestException as e:
            if attempt < retries - 1:
                print(f"⚠️ Erro na requisição ({e}). Tentativa {attempt+1}/{retries} em {delay} segundos...")
                time.sleep(delay)
                delay *= 2  # Backoff exponencial
            else:
                print(f"❌ Falha após {retries} tentativas para {url}")
                raise

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
                try:
                    with zf.open(info) as f:
                        yield info.filename, f.read()
                except Exception as e:
                    print(f"⚠️ Erro ao extrair {info.filename}: {e}")

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
                    on_bad_lines="skip"
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

def download_and_extract_for_year(session: requests.Session, year: int, out_raw: Path):
    print(f"⏳ Processando ano {year}...")
    try:
        res = get(session, BASE_URL)
    except Exception as e:
        print(f"🚫 Erro ao acessar página principal: {e}")
        return []

    try:
        year_links = find_year_links(res.text)
    except Exception as e:
        print(f"🚫 Erro ao encontrar links de anos: {e}")
        return []

    if year not in year_links:
        print(f"ℹ️ Nenhum link encontrado para o ano {year}.")
        return []

    year_url = year_links[year]
    print(f"🔗 URL do ano {year}: {year_url}")
    
    try:
        res_y = get(session, year_url)
    except Exception as e:
        print(f"🚫 Erro ao acessar página do ano {year}: {e}")
        return []

    content_type = res_y.headers.get("Content-Type", "").lower()
    if "text/html" not in content_type:
        print(f"📦[{year}] Link direto para arquivo detectado.")
        zip_links = [year_url]
    else:
        try:
            zip_links = find_zip_links(res_y.text)
            print(f"🔗 Encontrados {len(zip_links)} arquivos ZIP para {year}")
        except Exception as e:
            print(f"🚫[{year}] Erro ao parsear HTML: {e}")
            return []

    out_dfs = []
    for zurl in tqdm(zip_links, desc=f"{year} - zips"):
        try:
            print(f"⬇️ Baixando {zurl}")
            r = get(session, zurl, stream=True)
            content = r.content
            
            # Salva o arquivo ZIP bruto
            zip_filename = f"{year}_{os.path.basename(zurl)}"
            out_raw.mkdir(parents=True, exist_ok=True)
            zip_path = out_raw / zip_filename
            with open(zip_path, "wb") as f:
                f.write(content)
            print(f"💾 Arquivo salvo: {zip_path}")
            
            # Processa arquivos CSV dentro do ZIP
            csv_count = 0
            for fname, bytes_csv in iter_csv_from_zip(content):
                if is_target_station(fname):
                    print(f"🔍 CSV relevante encontrado: {fname}")
                    df = try_read_csv(bytes_csv)
                    if not df.empty:
                        out_dfs.append(normalize_columns(df))
                        csv_count += 1
                else:
                    print(f"❌ CSV ignorado: {fname} - não corresponde à estação alvo")
            
            print(f"📊 {csv_count} arquivos CSV processados para {year}")
        except Exception as e:
            print(f"🚫[{year}] Falha no arquivo {zurl}: {e}")
    
    return out_dfs

def get_neon_connection():
    try:
        # Adicionar verificação de variável de ambiente
        neon_url = os.getenv("NEON_DATABASE_URL")
        if not neon_url:
            print("🚫 Variável NEON_DATABASE_URL não configurada")
            return None
            
        # Forçar conexão SSL
        conn = psycopg2.connect(
            neon_url,
            sslmode="require"
        )
        return conn
    except Exception as e:
        print(f"🚫 Erro detalhado ao conectar ao NEON: {str(e)}")
        return None

def upload_to_neon(df: pd.DataFrame):
    if df.empty:
        print("ℹ️ Nenhum dado para enviar ao NEON")
        return

    # Garante colunas necessárias
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

    if not records:
        print("ℹ️ Nenhum registro válido para enviar")
        return

    conn = get_neon_connection()
    if not conn:
        return

    try:
        cur = conn.cursor()
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
        print(f"✅ {len(records)} registros inseridos/atualizados no NEON")
    except Exception as e:
        print(f"🚫 Erro ao enviar para o NEON: {e}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()

def check_inmet_availability():
    """Verifica se o site do INMET está respondendo"""
    try:
        test_session = requests.Session()
        test_session.headers.update({"User-Agent": "Mozilla/5.0"})
        response = test_session.get("https://portal.inmet.gov.br/", timeout=30)
        return response.status_code == 200
    except:
        return False

def main():
    print("🚀 Iniciando coleta de dados do INMET")
    
    # Verifica disponibilidade do INMET
    if not check_inmet_availability():
        print("🚨 O site do INMET não está respondendo. Abortando execução.")
        return

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
    # Rotaciona User-Agents para evitar bloqueio
    session.headers.update({"User-Agent": USER_AGENTS[datetime.now().second % len(USER_AGENTS)]})

    all_dfs = []
    for y in years:
        dfs = download_and_extract_for_year(session, y, Path(args.raw_dir) / str(y))
        all_dfs.extend(dfs)

    if not all_dfs:
        print("ℹ️ Nenhum dado encontrado para os anos solicitados")
        return

    # Filtra DataFrames válidos
    valid_dfs = [df for df in all_dfs if "DATA" in df.columns and not df.empty]
    if not valid_dfs:
        print("ℹ️ Nenhum dado válido com coluna DATA encontrado")
        return

    df_all = pd.concat(valid_dfs, ignore_index=True)
    df_all = df_all.dropna(subset=["DATA"]).sort_values("DATA").drop_duplicates(subset=["DATA"])
    
    print(f"📊 Total de registros processados: {len(df_all)}")
    upload_to_neon(df_all)
    print("✅ Processo concluído com sucesso!")

if __name__ == "__main__":
    main()
