from Import_And_Math import load_data_converte
from Import_And_Math import media_movel, adjust_offset, filter_low_pass
from Import_And_Math import filter_high_pass, indentify_outliers, remove_outliers
from Import_And_Math import DataStorage
import pandas as pd
import numpy as np
import dearpygui.dearpygui as dpg
import scipy as sp
from scipy import signal

"""Funções para processar os dados importados e plotar no gráfico.
    Contém as funções:
    processar_e_plotar: Processa os dados e plota no gráfico.
    callback_zomm: Função de callback para zoom no gráfico.
    select_archive: Função para selecionar o arquivo de dados."""

def processar_e_plotar(sender, app_data, user_data):

    if DataStorage.df_sensores.empty: 
        return

    df_trabalho = DataStorage.df_sensores.copy()

    if len(DataStorage.x_data) > 1:
        tempo_total = DataStorage.x_data[-1] - DataStorage.x_data[0]

        if tempo_total > 0:
            taxa_real = len(DataStorage.x_data) / tempo_total
        else:
            taxa_real = 1.0 
    else:
        taxa_real = 1.0

    print(f"Processando... Taxa: {taxa_real:.2f} Hz")


    # ---  Adicionando filtros um após o outro ---#
    
    n_offset = dpg.get_value("input_offset")
    if n_offset > 0:
        df_trabalho = adjust_offset(df_trabalho, n_offset)

    janela = dpg.get_value("input_janela_mm")
    if janela > 1:
        df_trabalho = media_movel(df_trabalho, janela)

    corte_low = dpg.get_value("input_passabaixa")
    if corte_low > 0 and corte_low < (taxa_real / 2):
        df_trabalho = filter_low_pass(df_trabalho, corte_low, sample_rate=taxa_real, order=2)   

    corte_high = dpg.get_value("input_highpass")
    if corte_high > 0 and corte_high < (taxa_real / 2):
        df_trabalho = filter_high_pass(df_trabalho, corte_high, freq_rate=taxa_real, order=2)

    remove_out = dpg.get_value("input_outliers")
    if remove_out > 0:
        df_trabalho = remove_outliers(df_trabalho, window=remove_out, thresh=3, verbose=False)

    # 3.3 ------ PLOTAGEM ---------
    dpg.delete_item("eixo_y", children_only=True)

    for col_name in DataStorage.colunas_disponiveis:
        tag_check = DataStorage.checkbox_tags.get(col_name)

        if tag_check and dpg.get_value(tag_check):
            if col_name in df_trabalho.columns:
                value_y = df_trabalho[col_name].tolist()
                dpg.add_line_series(DataStorage.x_data, value_y, parent="eixo_y", label=f"Sensor {col_name}")
    


    # 3.4 ------- Criando o zoom ------ #
def callback_zomm(sender, app_data):
    x_min, x_max = app_data[0], app_data[1]
    y_min, y_max = app_data[2], app_data[3]
    dpg.set_axis_limits("eixo_x", x_min, x_max)
    dpg.set_axis_limits("eixo_y", y_min, y_max)

# --------- Seleção de arquivo ----------- #

def select_archive(sender, app_data):
    #global x_data, df_sensores, colunas_disponiveis
    
    caminho = app_data['file_path_name']
    
    # 1. Carrega os novos dados
    DataStorage.x_data, DataStorage.df_sensores = load_data_converte(caminho, 0.00003375)
    
    if len(DataStorage.x_data) > 0:
        DataStorage.colunas_disponiveis = DataStorage.df_sensores.columns.tolist()
        
        # 2. Reconstrói a lista de Checkboxes
        dpg.delete_item("grupo_lista_canais", children_only=True)
        DataStorage.checkbox_tags.clear()
        
        for col in DataStorage.colunas_disponiveis:
            tag_chk = f"chk_{col}"
            DataStorage.checkbox_tags[col] = tag_chk
            # Marca os 3 primeiros por padrão
            estado = True if col in DataStorage.colunas_disponiveis[:3] else False
            dpg.add_checkbox(label=f"Sensor {col}", tag=tag_chk, default_value=estado, callback=processar_e_plotar, parent="grupo_lista_canais")
        
        processar_e_plotar(None, None, None)
        dpg.fit_axis_data("eixo_x")
        dpg.fit_axis_data("eixo_y")


