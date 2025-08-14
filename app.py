import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import psycopg2
from dotenv import load_dotenv

# Carrega variáveis de ambiente do arquivo .env (se existir)
load_dotenv()

st.set_page_config(page_title="Dashboard Meteorológico - São Luiz do Paraitinga", page_icon="🌤️", layout="wide")
st.title("🌤️ Dashboard Meteorológico - São Luiz do Paraitinga - SP")
st.caption("Fonte: INMET - Estações Automáticas")
st.markdown("---")

def get_neon_connection():
    """Cria e retorna uma conexão com o banco NEON."""
    try:
        conn = psycopg2.connect(os.getenv("NEON_DATABASE_URL"))
        return conn
    except Exception as e:
        st.error(f"Erro ao conectar ao NEON: {e}")
        return None

@st.cache_data(show_spinner=False)
def fetch_from_neon():
    """Busca todos os dados da tabela inmet_data."""
    conn = get_neon_connection()
    if not conn:
        return pd.DataFrame(columns=["data"])
    try:
        query = """
        SELECT 
            data, 
            temperatura_media, 
            temperatura_maxima, 
            temperatura_minima, 
            umidade_relativa, 
            precipitacao, 
            velocidade_vento, 
            pressao_atmosferica 
        FROM inmet_data;
        """
        df = pd.read_sql(query, conn)
        # Renomear colunas para o formato original (em maiúsculas e com os nomes anteriores)
        df.rename(columns={
            "data": "DATA",
            "temperatura_media": "TEMPERATURA_MEDIA",
            "temperatura_maxima": "TEMPERATURA_MAXIMA",
            "temperatura_minima": "TEMPERATURA_MINIMA",
            "umidade_relativa": "UMIDADE_RELATIVA",
            "precipitacao": "PRECIPITACAO",
            "velocidade_vento": "VELOCIDADE_VENTO",
            "pressao_atmosferica": "PRESSAO_ATMOSFERICA"
        }, inplace=True)
        df["DATA"] = pd.to_datetime(df["DATA"])
        return df.sort_values("DATA")
    except Exception as e:
        st.error(f"Erro ao buscar dados: {e}")
        return pd.DataFrame(columns=["DATA"])
    finally:
        if conn:
            conn.close()

# Carrega os dados do NEON
df = fetch_from_neon()

# Se não há dados, exibe um aviso e para a execução
if df.empty:
    st.warning("Não há dados disponíveis no banco. Por favor, rode o script de coleta (`fetch_inmet.py`) ou aguarde a atualização.")
    st.stop()

# Filtros na sidebar
st.sidebar.header("🔧 Filtros")
min_date, max_date = df["DATA"].min().date(), df["DATA"].max().date()
rng = st.sidebar.date_input("Período", value=(min_date, max_date), min_value=min_date, max_value=max_date)

# Ajuste para o caso de o usuário selecionar apenas uma data (em vez de um intervalo)
if isinstance(rng, (list, tuple)) and len(rng) == 2:
    start, end = rng
    dff = df[(df["DATA"].dt.date >= start) & (df["DATA"].dt.date <= end)].copy()
else:
    dff = df.copy()

# Métricas Principais
st.header("📊 Métricas Principais")
c1, c2, c3, c4 = st.columns(4)
with c1:
    val = dff["TEMPERATURA_MEDIA"].mean()
    st.metric("Temperatura média", f"{val:.1f} °C" if pd.notna(val) else "—")
with c2:
    val = dff["UMIDADE_RELATIVA"].mean()
    st.metric("Umidade média", f"{val:.1f} %" if pd.notna(val) else "—")
with c3:
    val = dff["PRECIPITACAO"].sum()
    st.metric("Precipitação total", f"{val:.1f} mm" if pd.notna(val) else "—")
with c4:
    val = dff["VELOCIDADE_VENTO"].mean()
    st.metric("Velocidade do vento", f"{val:.1f} m/s" if pd.notna(val) else "—")

st.markdown("---")
st.header("📈 Análise Temporal")

# Gráfico de temperaturas
fig_temp = px.line(dff, x="DATA", y=["TEMPERATURA_MAXIMA","TEMPERATURA_MINIMA","TEMPERATURA_MEDIA"],
                   labels={"value":"Temperatura (°C)", "variable":"Série"}, title="Temperaturas ao longo do tempo")
st.plotly_chart(fig_temp, use_container_width=True)

# Gráfico de precipitação
fig_p = px.bar(dff, x="DATA", y="PRECIPITACAO", title="Precipitação diária (mm)")
st.plotly_chart(fig_p, use_container_width=True)

# Gráfico de umidade
fig_u = px.line(dff, x="DATA", y="UMIDADE_RELATIVA", title="Umidade relativa (%)")
st.plotly_chart(fig_u, use_container_width=True)

st.markdown("---")
st.header("🤖 Aprendizado de Máquina (exemplo)")
st.caption("Regressão Linear para estimar Temperatura Média usando mês e dia do ano.")

# Verifica se há dados suficientes para treinar o modelo
if dff["TEMPERATURA_MEDIA"].notna().sum() > 30:
    df_ml = dff[["DATA","TEMPERATURA_MEDIA"]].dropna().copy()
    df_ml["MES"] = df_ml["DATA"].dt.month
    df_ml["DIA_DO_ANO"] = df_ml["DATA"].dt.dayofyear
    X = df_ml[["MES","DIA_DO_ANO"]]
    y = df_ml["TEMPERATURA_MEDIA"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression().fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    c1, c2 = st.columns(2)
    with c1:
        st.metric("MSE", f"{mse:.3f}")
        st.metric("R²", f"{r2:.3f}")
    with c2:
        fig_ml = go.Figure()
        fig_ml.add_trace(go.Scatter(x=y_test, y=y_pred, mode="markers", name="Predições"))
        lo = float(min(y_test.min(), y_pred.min()))
        hi = float(max(y_test.max(), y_pred.max()))
        fig_ml.add_trace(go.Scatter(x=[lo,hi], y=[lo,hi], mode="lines", name="Perfeito", line=dict(dash="dash")))
        fig_ml.update_layout(title="Predição vs. Real (Temp. Média)",
                             xaxis_title="Real (°C)", yaxis_title="Predito (°C)")
        st.plotly_chart(fig_ml, use_container_width=True)
else:
    st.info("Ainda não há dados suficientes de temperatura média para treinar o modelo.")

st.markdown("---")
st.header("📄 Dados brutos")
if st.checkbox("Mostrar dados filtrados"):
    st.dataframe(dff)

# Botão para baixar dados filtrados
st.download_button(
    label="📥 Baixar CSV filtrado",
    data=dff.to_csv(index=False).encode("utf-8"),
    file_name="dados_filtrados.csv",
    mime="text/csv"
)
