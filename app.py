# app.py - VERSÃO CORRIGIDA E COMPLETA

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
# psycopg2 e dotenv não são mais necessários aqui, mas podem ser mantidos
# import psycopg2
# from dotenv import load_dotenv
import time

# load_dotenv() # Não é mais necessário no Streamlit Cloud com Secrets

st.set_page_config(page_title="Dashboard Meteorológico - São Luiz do Paraitinga", page_icon="🌤️", layout="wide")
st.title("🌤️ Dashboard Meteorológico - São Luiz do Paraitinga - SP")
st.caption("Fonte: INMET - Estações Automáticas")
st.markdown("---")

# Modificação 1: Adicionar TTL de 6 horas no cache
@st.cache_data(show_spinner=False, ttl=6*3600)  # Cache de 6 horas
def fetch_from_neon():
    """Busca todos os dados da tabela inmet_data usando st.connection."""
    with st.spinner("Carregando dados do banco..."):
        try:
            # Conecta usando a configuração de [connections.neon_db] dos Secrets
            conn = st.connection("neon_db", type="sql")
            
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
            df = conn.query(query)
            
            # Renomear colunas para o formato original
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
            st.error(f"Erro ao buscar dados com st.connection: {e}")
            return pd.DataFrame(columns=["DATA"])

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

# Ajuste para o caso de o usuário selecionar apenas uma data
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
st.header("🤖 Modelo Preditivo")
st.caption("Previsão de temperatura média usando dados históricos")

# Inicializa variáveis para configuração
features = ["MES", "DIA_DO_ANO"]
model_type = "Regressão Linear"
test_size = 0.2

# Interface para configuração do modelo
with st.expander("🔧 Configurações do Modelo", expanded=False):
    features = st.multiselect("Variáveis preditoras", 
                             ["MES", "DIA_DO_ANO", "UMIDADE_RELATIVA", "PRECIPITACAO"],
                             default=["MES", "DIA_DO_ANO"])
    model_type = st.selectbox("Algoritmo", 
                             ["Regressão Linear", "Random Forest", "Gradient Boosting"])
    test_size = st.slider("Tamanho do conjunto de teste", 0.1, 0.5, 0.2, 0.05)

# Verifica se há dados suficientes para treinar o modelo
if dff["TEMPERATURA_MEDIA"].notna().sum() > 100:
    df_ml = dff[["DATA", "TEMPERATURA_MEDIA", "UMIDADE_RELATIVA", "PRECIPITACAO"]].dropna().copy()
    df_ml["MES"] = df_ml["DATA"].dt.month
    df_ml["DIA_DO_ANO"] = df_ml["DATA"].dt.dayofyear
    
    if not features:
        st.error("Selecione pelo menos uma variável preditora.")
        st.stop()
    
    X = df_ml[features]
    y = df_ml["TEMPERATURA_MEDIA"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    if model_type == "Regressão Linear":
        model = LinearRegression()
    elif model_type == "Random Forest":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        model = GradientBoostingRegressor(random_state=42)
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    st.subheader("Resultados do Modelo")
    col1, col2, col3 = st.columns(3)
    col1.metric("R²", f"{r2:.3f}")
    col2.metric("MAE", f"{mae:.3f} °C")
    col3.metric("MSE", f"{mse:.3f}")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers', name='Predições', marker=dict(color='#2878B5')))
    lo, hi = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode='lines', name='Perfeito', line=dict(color='red', dash='dash')))
    fig.update_layout(title='Valores Reais vs Preditos', xaxis_title='Temperatura Real (°C)', yaxis_title='Temperatura Predita (°C)')
    st.plotly_chart(fig, use_container_width=True)
    
    if hasattr(model, 'feature_importances_'):
        st.subheader("Importância das Variáveis")
        importances = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_}).sort_values('Importance', ascending=False)
        fig_imp = px.bar(importances, x='Importance', y='Feature', orientation='h', title='Importância das características', color='Importance')
        st.plotly_chart(fig_imp, use_container_width=True)
else:
    st.warning("Dados insuficientes para treinar o modelo. São necessários pelo menos 100 registros de temperatura média.")

st.markdown("---")
st.header("📄 Dados brutos")
if st.checkbox("Mostrar dados filtrados"):
    st.dataframe(dff)

st.download_button(
    label="📥 Baixar CSV filtrado",
    data=dff.to_csv(index=False).encode("utf-8"),
    file_name="dados_filtrados.csv",
    mime="text/csv"
)
