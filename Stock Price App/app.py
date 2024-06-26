import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import yfinance as yf
import base64
import streamlit as st

st.title("Stock Price App")

st.markdown("""
This app retreives the list of the **Stocks** and its corresponding **Stock Closing Price** (year-to-date)         
* **Python Libraries:** base64, stramlit, yfinance, seaborn, numpy, pandas, matplotlib
* **Data Source:** [Wikipedia](https://en.wikipedia.org/wiki/List_of_S%26P_500_companies)
""")

st.sidebar.header("User Input Features")

@st.cache
def load_data():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    html = pd.read_html(url, header=0)
    df = html[0]
    return df

df = load_data()
sector = df.groupby('GICS Sector')

sorted_sector_unique = sorted( df['GICS Sector'].unique() )
selected_sector = st.sidebar.multiselect('Sector', sorted_sector_unique, sorted_sector_unique)

df_selected_sector = df[(df['GICS Sector'].isin(selected_sector))]

st.header("Display Companies in Selected Sector")
st.write("Data Dimension: " + str(df_selected_sector.shape[0]) + " rows and " + str(df_selected_sector.shape[1]) + " columns.")
st.dataframe(df_selected_sector)

def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="SP500.csv">Download CSV File</a>'
    return href

st.markdown(filedownload(df_selected_sector), unsafe_allow_html=True)

data = yf.download(
    tickers = list(df_selected_sector[:10].Symbol),
    period = "ytd",
    interval = "1d",
    group_by = 'ticker',
    auto_adjust = True,
    prepost = True,
    threads = True,
    proxy = None
)

def price_plot(symbol):
    df = pd.DataFrame(data[symbol].Close)
    df['Date'] = df.index
    plt.fill_between(df.Date, df.Close, color = 'skyblue', alpha=0.3)
    plt.plot(df.Date, df.Close, color='skyblue', alpha=0.3)
    plt.xticks(rotation=90)
    plt.title(symbol, fontweight='bold')
    plt.xlabel('Date', fontweight='bold')
    plt.ylabel('Closing Price', fontweight='bold')
    return st.pyplot()

num_company = st.sidebar.slider("Number of Companies", 1, 5)

if st.button("Show Plots"):
    st.header("Stock Closing Price")
    for i in list(df_selected_sector.Symbol)[:num_company]:
        price_plot(i)