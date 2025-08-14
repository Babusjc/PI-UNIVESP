import os, re, io, zipfile, argparse, unicodedata, time, traceback
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
import numpy as np

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
    name = re.sub(r"[^a-z0-9]+", " ", name)
    name = re.sub(r"\bsao\b", "são", name)
    name = re.sub(r"\bluis\b", "luiz", name)
    return name.strip()

def is_target_station(filename: str) -> bool:
    """Verifica se o arquivo é da estação desejada."""
    normalized = normalize_station_name(filename)
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
                delay *= 2
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

def preprocess_csv(content: bytes) -> str:
    """Pré-processa o conteúdo CSV removendo metadados e corrigindo formatos"""
    try:
        # Decodifica usando latin-1 que é comum em dados brasileiros
        content_str = content.decode('latin-1')
    except UnicodeDecodeError:
        try:
            content_str = content.decode('utf-8')
        except:
            content_str = content.decode('latin-1', errors='replace')
    
    lines = content_str.splitlines()
    clean_lines = []
    data_started = False
    
    # Padrões para identificar início dos dados
    header_patterns = [
        "PRECIPITA", "PRESSAO", "RADIACAO", "TEMPERAT", "UMIDADE", "VENTO"
    ]
    
    for line in lines:
        # Verifica se encontrou o cabeçalho de dados
        if any(pattern in line for pattern in header_patterns):
            data_started = True
            
        if data_started:
            # Corrige decimais (substitui vírgula por ponto)
            line = re.sub(r'(\d+),(\d+)', r'\1.\2', line)
            # Remove caracteres problemáticos
            line = line.replace('。', '').replace('‘', '').replace('(', '')
            clean_lines.append(line)
    
    return "\n".join(clean_lines)

def try_read_csv(bytes_content: bytes) -> pd.DataFrame:
    """Tenta ler o CSV com tratamento avançado de pré-processamento"""
    try:
        # Pré-processa o conteúdo
        clean_csv = preprocess_csv(bytes_content)
        
        # Tenta ler com diferentes parâmetros
        try:
            return pd.read_csv(
                io.StringIO(clean_csv),
                sep=";",
                skip_blank_lines=True,
                engine="python",
                on_bad_lines="warn",
                decimal=".",
                thousands=None
            )
        except:
            return pd.read_csv(
                io.StringIO(clean_csv),
                sep=",",
                skip_blank_lines=True,
                engine="python",
                on_bad_lines="warn",
                decimal=".",
                thousands=None
            )
    except Exception as e:
        print(f"⚠️ Falha crítica na leitura do CSV: {e}")
        return pd.DataFrame()

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza nomes de colunas e converte tipos de dados"""
    def norm(c):
        c2 = unicodedata.normalize("NFKD", str(c)).encode("ascii", "ignore").decode("ascii").upper()
        c2 = re.sub(r"[^A-Z0-9]+", "_", c2).strip("_")
        return c2
    
    df = df.rename(columns={c: norm(c) for c in df.columns})
    
    # Mapeamento de colunas alternativas
    column_mapping = {
        "DATA": ["DATA", "DATE"],
        "HORA_UTC": ["HORA_UTC", "HORA"],
        "TEMPERATURA_MEDIA": ["TEMPERATURA_MEDIA", "TEMP_MED", "TEMP"],
        "UMIDADE_RELATIVA": ["UMIDADE", "UMID_REL"],
        "PRECIPITACAO": ["PRECIPITA", "CHUVA"],
        "PRESSAO_ATMOSFERICA": ["PRESSAO", "PRESSAO_ATM"]
    }
    
    # Renomeia colunas alternativas
    for standard, alternates in column_mapping.items():
        for alt in alternates:
            if alt in df.columns and standard not in df.columns:
                df[standard] = df[alt]
    
    # Conversão de tipos de dados
    if "DATA" in df.columns:
        try:
            df["DATA"] = pd.to_datetime(df["DATA"], errors="coerce", dayfirst=True)
        except:
            pass
    
    # Converte colunas numéricas
    numeric_cols = [col for col in df.columns if any(kw in col for kw in 
                   ["TEMP", "UMID", "PREC", "PRES", "RAD", "VENTO"])]
    
    for col in numeric_cols:
        try:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        except:
            pass
    
    return df

def download_and_extract_for_year(session: requests.Session, year: int, out_raw: Path):
    print(f"⏳ Processando ano {year}...")
    try:
        res = get(session, BASE_URL)
        year_links = find_year_links(res.text)
    except Exception as e:
        print(f"🚫 Erro ao acessar página principal: {e}")
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
            
            # Processa arquivos CSV
            csv_count = 0
            for fname, bytes_csv in iter_csv_from_zip(content):
                if is_target_station(fname):
                    print(f"🔍 CSV relevante encontrado: {fname}")
                    df = try_read_csv(bytes_csv)
                    if not df.empty:
                        df = normalize_columns(df)
                        if "DATA" in df.columns:
                            out_dfs.append(df)
                            csv_count += 1
                            print(f"✅ Dados válidos encontrados: {len(df)} registros")
                else:
                    print(f"❌ CSV ignorado: {fname} - não corresponde à estação alvo")
            
            print(f"📊 {csv_count} arquivos CSV processados para {year}")
        except Exception as e:
            print(f"🚫[{year}] Falha no arquivo {zurl}: {e}")
            traceback.print_exc()
    
    return out_dfs

def get_neon_connection():
    """Estabelece conexão com o banco NEON com verificações detalhadas"""
    try:
        neon_url = os.getenv("NEON_DATABASE_URL")
        if not neon_url:
            print("🚫 Variável NEON_DATABASE_URL não configurada")
            return None
            
        # Verificação detalhada da URL
        if "postgresql://" not in neon_url:
            print("⚠️ Formato inválido da URL do NEON. Deve começar com postgresql://")
            return None
            
        # Teste de conexão
        conn = psycopg2.connect(neon_url, sslmode="require")
        
        # Teste de consulta simples
        cur = conn.cursor()
        cur.execute("SELECT 1")
        result = cur.fetchone()
        
        if result and result[0] == 1:
            print("✅ Conexão com NEON verificada com sucesso")
            return conn
        else:
            print("⚠️ Teste de conexão com NEON falhou")
            return None
            
    except psycopg2.OperationalError as oe:
        print(f"🚫 Erro operacional ao conectar ao NEON: {oe}")
        print("Verifique: 1) URL correta 2) Permissões 3) Firewall 4) Status do serviço")
    except Exception as e:
        print(f"🚫 Erro inesperado ao conectar ao NEON: {e}")
    
    return None

def upload_to_neon(df: pd.DataFrame):
    """Envia dados para o banco NEON com tratamento robusto"""
    if df.empty:
        print("ℹ️ Nenhum dado para enviar ao NEON")
        return

    # Verifica colunas mínimas necessárias
    required_columns = ["DATA"]
    for col in required_columns:
        if col not in df.columns:
            print(f"🚫 Coluna obrigatória '{col}' não encontrada nos dados")
            return

    # Mapeamento de colunas para a estrutura do banco
    column_mapping = {
        "data": "DATA",
        "temperatura_media": ["TEMPERATURA_MEDIA", "TEMP_MEDIA"],
        "temperatura_maxima": ["TEMPERATURA_MAXIMA", "TEMP_MAX"],
        "temperatura_minima": ["TEMPERATURA_MINIMA", "TEMP_MIN"],
        "umidade_relativa": ["UMIDADE_RELATIVA", "UMIDADE"],
        "precipitacao": ["PRECIPITACAO", "CHUVA"],
        "velocidade_vento": ["VELOCIDADE_VENTO", "VENTO"],
        "pressao_atmosferica": ["PRESSAO_ATMOSFERICA", "PRESSAO"]
    }
    
    # Prepara DataFrame para inserção
    output_df = pd.DataFrame()
    output_df["data"] = pd.to_datetime(df["DATA"], errors="coerce")
    
    # Adiciona colunas opcionais
    for neon_col, source_cols in column_mapping.items():
        if neon_col == "data":
            continue
            
        for src in source_cols:
            if src in df.columns:
                output_df[neon_col] = pd.to_numeric(df[src], errors="coerce")
                break
        else:
            output_df[neon_col] = None
    
    # Remove registros sem data válida
    output_df = output_df.dropna(subset=["data"])
    
    if output_df.empty:
        print("ℹ️ Nenhum registro válido após preparação")
        return

    # Converte para lista de tuplas
    records = output_df.to_records(index=False).tolist()
    print(f"📤 Preparados {len(records)} registros para envio ao NEON")

    conn = get_neon_connection()
    if not conn:
        return

    try:
        cur = conn.cursor()
        
        # Verifica se a tabela existe
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'inmet_data'
            );
        """)
        table_exists = cur.fetchone()[0]
        
        if not table_exists:
            print("🚫 Tabela 'inmet_data' não existe no banco de dados")
            print("Execute este comando SQL para criar a tabela:")
            print("""
                CREATE TABLE inmet_data (
                    id SERIAL PRIMARY KEY,
                    data TIMESTAMP UNIQUE,
                    temperatura_media FLOAT,
                    temperatura_maxima FLOAT,
                    temperatura_minima FLOAT,
                    umidade_relativa FLOAT,
                    precipitacao FLOAT,
                    velocidade_vento FLOAT,
                    pressao_atmosferica FLOAT
                );
            """)
            return
        
        # Query de inserção
        query = sql.SQL("""
            INSERT INTO inmet_data (
                data, temperatura_media, temperatura_maxima, temperatura_minima,
                umidade_relativa, precipitacao, velocidade_vento, pressao_atmosferica
            ) VALUES %s
            ON CONFLICT (data) DO UPDATE SET
                temperatura_media = EXCLUDED.temperatura_media,
                temperatura_maxima = EXCLUDED.temperatura_maxima,
                temperatura_minima = EXCLUDED.temperatura_minima,
                umidade_relativa = EXCLUDED.umidade_relativa,
                precipitacao = EXCLUDED.precipitacao,
                velocidade_vento = EXCLUDED.velocidade_vento,
                pressao_atmosferica = EXCLUDED.pressao_atmosferica;
        """)
        
        execute_values(
            cur, 
            query, 
            records,
            template="(TIMESTAMP %s, %s, %s, %s, %s, %s, %s, %s)"
        )
        
        conn.commit()
        print(f"✅ {len(records)} registros inseridos/atualizados no NEON")
        
    except psycopg2.Error as e:
        print(f"🚫 Erro PostgreSQL: {e.pgerror}")
        print(f"Código: {e.pgcode}")
        conn.rollback()
    except Exception as e:
        print(f"🚫 Erro inesperado: {e}")
        traceback.print_exc()
        conn.rollback()
    finally:
        if 'cur' in locals():
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
    session.headers.update({"User-Agent": USER_AGENTS[datetime.now().second % len(USER_AGENTS)]})

    all_dfs = []
    for y in years:
        dfs = download_and_extract_for_year(session, y, Path(args.raw_dir) / str(y))
        all_dfs.extend(dfs)

    if not all_dfs:
        print("ℹ️ Nenhum dado encontrado para os anos solicitados")
        return

    valid_dfs = [df for df in all_dfs if "DATA" in df.columns and not df.empty]
    if not valid_dfs:
        print("ℹ️ Nenhum dado válido com coluna DATA encontrado")
        return

    df_all = pd.concat(valid_dfs, ignore_index=True)
    df_all = df_all.dropna(subset=["DATA"]).sort_values("DATA")
    
    # Remover duplicatas mantendo o último registro
    df_all = df_all.drop_duplicates(subset=["DATA"], keep="last")
    
    print(f"📊 Total de registros processados: {len(df_all)}")
    upload_to_neon(df_all)
    print("✅ Processo concluído!")

if __name__ == "__main__":
    main()
