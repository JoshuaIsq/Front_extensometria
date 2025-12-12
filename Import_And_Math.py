import pandas as pd
import numpy as np
import scipy as sp
from scipy import signal

"""Abre arquivo de medição para análise experimental de tensões
    Contém funções para carregar dados de arquivos .txt (necessário incluir outros futuramente),
    possui as funções:
    load_data_converte: Carrega e converte os dados do arquivo."""

class DataStorage:
    x_data = []
    df_sensores = pd.DataFrame()
    checkbox_tags = {} 
    colunas_disponiveis = []



def load_data_converte(filename, calibration):

    passo = 1
    if filename.endswith(".txt"):
        try:

            txt_file = pd.read_csv(filename, sep=r'[;\s]+', header=None, engine="python", on_bad_lines="skip") 

            cols_tempo_indices = [0, 1, 2, 3, 4, 5]
            for col in cols_tempo_indices:
                txt_file[col] = pd.to_numeric(txt_file[col], errors='coerce')
            
            txt_file = txt_file.dropna(subset=cols_tempo_indices)
            txt_file[cols_tempo_indices] = txt_file[cols_tempo_indices].astype(int)
                
            #Fazer uma limpeza organizacional deste trecho futuramente
            time_cols = txt_file.iloc[:, 0:6] 
            time_cols.columns = ["day", "month", "year", "hour", "minute", "second"]
            timestamp = pd.to_datetime(time_cols)
            df_temp = pd.DataFrame({'timestamp': timestamp})
            df_data = pd.concat([df_temp, txt_file.iloc[:, 7:]], axis=1)
            df_sorted = df_data.sort_values(by='timestamp').reset_index(drop=True)
            start_time = df_sorted['timestamp'].iloc[0]
            #Função que altera o eixo tempo (pode ser segundos, minutos, horas, dias, sujeito a manipulação)
            eixo_x_segundos = ((df_sorted['timestamp'] - start_time).dt.total_seconds()).tolist() 
            sensores_df = df_sorted.iloc[:, 1:].fillna(0) * calibration 
            sensores_df = sensores_df.interpolate(method='linear', limit_direction='both').fillna(0)
            sensores_df.columns = [str(c) for c in sensores_df.columns]
            amount_data = len(sensores_df.columns)
            new_name = [str(i + 1) for i in range(amount_data)]
            sensores_df.columns = new_name

            print(f"Sucesso! {len(eixo_x_segundos)} pontos, {len(sensores_df.columns)} canais.")
            return eixo_x_segundos, sensores_df
            
        except Exception as e:
            print(f"Erro ao carregar: {e}")
            return [], pd.DataFrame()
            
    return [], pd.DataFrame()


#2. ---------------- Filtros e ajustes -------------

"""Este trecho contem todas as funções de filtros e ajustes que podem ser aplicados
    Média move: Suaviza os dados aplicando uma média móvel com janela definida pelo usuário.
    Ajuste de offset: Remove o offset inicial dos dados com base na média dos primeiros n pontos.
    Filtro passa baixa: Aplica um filtro Butterworth passa baixa para remover ruídos de alta frequência.
    Filtro passa alta: Aplica um filtro Butterworth passa alta para remover tendências de baixa frequência.
    Identificação de outliers: Detecta pontos fora do padrão usando z-score baseado em média móvel.
    Remoção de outliers: Substitui os outliers identificados por interpolação linear.
    """

def media_movel(df, janela):
    df_copia = df.copy() 
    df_copia = df_copia.rolling(window=int(janela), min_periods=1).mean() 

    return df_copia.round(4)

def adjust_offset(df, n_linhas):
    df_copia = df.copy()
    adjust = df_copia.iloc[:int(n_linhas)].mean()
    df_copia = df_copia - adjust

    return df_copia


def filter_low_pass(df, cut_freq, sample_rate, order):
    df_copia = df.copy()
    nyquisfreq = 0.5 * sample_rate
    low_pass_ratio = cut_freq/nyquisfreq
    b, a = signal.butter(order, low_pass_ratio, btype="lowpass")
    for col in df_copia.columns:
        df_copia[col] = signal.filtfilt(b, a, df_copia[col])

    return df_copia.round(4)


def filter_high_pass(df, freq_corte, freq_rate, order):
    df_copia = df.copy()
    nyquisfreq = 0.5 * freq_rate
    filter_high_pass = freq_corte/nyquisfreq
    b, a = signal.butter(order, filter_high_pass, btype="highpass")
    for col in df_copia.columns:
        df_copia[col] = signal.filtfilt(b, a, df_copia[col])

    return df_copia.round(4)


def indentify_outliers(df, window, thresh=3, verbose=False):
    df_copia = df.copy()
    outlier_mask = pd.DataFrame(False, index=df_copia.index, columns=df_copia.columns)
    for col in df_copia.columns:
        series = df_copia[col]
        rolling_mean = series.rolling(window=window, min_periods=1).mean()
        rolling_std = series.rolling(window=window, min_periods=1).std()
        z_scores = (series - rolling_mean) / rolling_std
        outliers = np.abs(z_scores) > thresh
        outlier_mask[col] = outliers
        if verbose:
                print(f"[INFO] Coluna: {col}")
                print(f"       Média: {series.mean():.2f}, Desvio padrão: {series.std():.2f}")
                print(f"       Outliers detectados: {outliers.sum()} de {len(series)}\n")

    return outlier_mask
    
def remove_outliers(df, window, thresh=3, verbose=False):
    df_copia = df.copy()
    outlier_mask = indentify_outliers(df_copia, window, thresh, verbose)
    df_copia = df_copia.mask(outlier_mask)
    df_copia = df_copia.interpolate(method='linear', limit_direction='both').fillna(0)

    return df_copia.round(4)
