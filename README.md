# Dashboard Meteorológico – São Luiz do Paraitinga (INMET)

Pipeline completo: coleta no portal do **INMET**, filtragem da estação **São Luiz do Paraitinga**, combinação dos CSVs e visualização em **Streamlit** com exemplo de ML.

## Como usar
1) Instale dependências:
```bash
pip install -r requirements.txt
```
2) Baixe os dados (todos os anos):
```bash
python fetch_inmet.py --years all --raw_dir data/raw --combined data/inmet_data_sao_luiz_do_paraitinga_combined.csv
```
3) Rode o dashboard:
```bash
streamlit run app.py
```

## Deploy no Streamlit Cloud
- Suba o repositório para o GitHub e aponte o app para `app.py`.
- Para atualização automática mensal, use o workflow em `.github/workflows/update_inmet.yml`.

