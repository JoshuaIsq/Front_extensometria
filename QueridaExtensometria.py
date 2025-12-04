import pandas as pd
import numpy as np
import dearpygui.dearpygui as dpg
import scipy as sp
from scipy import signal

#1 ---------------- Carregar dados do arquivo ------------------ #
def load_data_converte(filename, calibration):

    passo = 1
    if filename.endswith(".txt"):
        #1.1 ------ Importando o meu arquivo txt com leitura extensometrica --------

        txt_file = pd.read_csv(filename, sep=r'[;\s]+', header=None, engine="python") 

        #1.2 ---- Transformando possiveis linhas problematicas em linhas funcionais -----

        # Vamos garantir que as primeiras 6 colunas (tempo) sejam NÚMEROS
        # errors='coerce' transforma tudo que for texto ruim/vazio em NaN (Not a Number)
        cols_tempo_indices = [0, 1, 2, 3, 4, 5]
        for col in cols_tempo_indices:
            txt_file[col] = pd.to_numeric(txt_file[col], errors='coerce')
        
        #1.3 --- Descartando linhas que ficaram em NaN
    
        txt_file = txt_file.dropna(subset=cols_tempo_indices)
        # Converte para inteiro (porque dia 10.0 fica feio, queremos dia 10)
        txt_file[cols_tempo_indices] = txt_file[cols_tempo_indices].astype(int)
            
       
        #1.4 ------ Criando timestamp das horas --------
        time_cols = txt_file.iloc[:, 0:6] #Colocando o timestamp e as colunas de tempo
        time_cols.columns = ["day", "month", "year", "hour", "minute", "second"]
        timestamp = pd.to_datetime(time_cols)
        df_temp = pd.DataFrame({'timestamp': timestamp})
        df_data = pd.concat([df_temp, txt_file.iloc[:, 7:]], axis=1)
        df_sorted = df_data.sort_values(by='timestamp').reset_index(drop=True)
        start_time = df_sorted['timestamp'].iloc[0]
        eixo_x_segundos = (df_sorted['timestamp'] - start_time).dt.total_seconds().tolist()
        eixo_x_segundos = eixo_x_segundos[::50]
        sensores_df = df_sorted.iloc[:, 1:].fillna(0) * calibration
        sensores_df = sensores_df.iloc[::50, :]

        return eixo_x_segundos, sensores_df
        #Ao final a função me retorna o eixo X que mostra quanto tempo passou
        #E sensores_df me respresenta meus sensores ja recebendo o valor de calibração dados
        #Quando eu chamar a função, eu passo pra ela os dados de 
    
    return [], pd.DataFrame()



#2. ---------------- Filtros e ajustes -------------

#2.1 ------ Média movel ------- 

def media_movel(df, janela):###
    df_copia = df.copy() #criando uma copia do meu dataframe pra não quebrar ele
    df_copia = df_copia.rolling(window=int(janela), min_periods=1).mean() #aplicação do filtro de fato atráves do pandas
    return df_copia.round(4)

#2.2 ------ Ajuste de offset ------- 

def adjust_offset(df, n_linhas):
    df_copia = df.copy()
    adjust = df_copia.iloc[:int(n_linhas)].mean()
    df_copia = df_copia - adjust

    return df_copia

#2.3 ------ Filtro Passa Baixa ------- 

def filter_low_pass(df, cut_freq, sample_rate, order):
    df_copia = df.copy()
    nyquisfreq = 0.5 * sample_rate
    low_pass_ratio = cut_freq/nyquisfreq
    b, a = signal.butter(order, low_pass_ratio, btype="low")
    for col in df_copia.columns:
        df_copia[col] = signal.filtfilt(b, a, df_copia[col])
    return df_copia.round(4)



# ---------- Criação d botões ----------

def callback_botao_passabaixa():
    valor_digi = dpg.get_value("input_passabaixa")
    df_filtrado = filter_low_pass(df_sensores, valor_digi, sample_rate=25000, order=2)
    colunas = df_filtrado.columns
    for i in range(min(18, len(colunas))):
        col_name = colunas[i]
        y_novo = df_filtrado[col_name].tolist()
        dpg.set_value(f"serie_canal_{i}", [x_data, y_novo])

def callback_botao_filtro():
    # 1. Pega o valor que você digitou na caixinha
    valor_janela = dpg.get_value("input_janela_mm")
    df_filtrado = media_movel(df_sensores, valor_janela)
    colunas = df_filtrado.columns
    for i in range(min(18, len(colunas))):
        col_name = colunas[i]
        y_novo = df_filtrado[col_name].tolist()
        dpg.set_value(f"serie_canal_{i}", [x_data, y_novo])
    
    # 2. Aplica a matemática (note que usamos suas variáveis originais)
    # Guardamos o resultado numa nova variável para não perder o original
    #df_filtrado = media_movel(df_sensores, valor_janela)
    
    # Por enquanto, vamos apenas mostrar no terminal para provar que funcionou
    print(f"Filtro aplicado com janela {valor_janela}!")
    print(df_filtrado.head()) 

def callback_botao_offset():
    n_linhas = dpg.get_value("input_offset")
    df_offset = adjust_offset(df_sensores, n_linhas)
    colunas = df_offset.columns
    for i in range(min(18, len(colunas))):
        col_name = colunas[i]
        y_novo = df_offset[col_name].tolist()
        dpg.set_value(f"serie_canal_{i}", [x_data, y_novo])


x_data, df_sensores = load_data_converte("LOG_1.txt", 0.00003375)



#3. --------------- Interface -------------------- #
dpg.create_context()

#3.1 --------- Cor das janelas --------#

with dpg.theme() as white_color:
    with dpg.theme_component(dpg.mvWindowAppItem):
        dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (255, 255, 255, 255))

#3.2 ----- Janelas principais ------#

with dpg.window(tag="Primary Window"):
    dpg.add_text("VISUALIZADOR DE EXTENSOMETRIA", color=(0, 255, 255))

    #3.2.1 ---- Botões -----

    with dpg.group(horizontal=True):
        # Onde você digita o valor da média
        dpg.add_input_int(default_value=10, width=150, tag="input_janela_mm")
        
        # O botão que chama a função que criamos acima
        dpg.add_button(label="Aplicar Média Móvel", callback=callback_botao_filtro)


    dpg.add_separator()

    dpg.add_text("Controle de Offset (Zerar):")
    
    with dpg.group(horizontal=True):
        # Input para definir quantas linhas usar (Tag importante!)
        dpg.add_input_int(default_value=10, width=150, tag="input_offset", min_value=1)
        
        # O Botão que chama a função de offset
        dpg.add_button(label="Aplicar Offset", callback=callback_botao_offset)

    dpg.add_separator()

    dpg.add_text("Frequecia de corte")

    with dpg.group(horizontal=True):
        dpg.add_input_int(default_value=10, width=150, tag='input_passabaixa', min_value=1)

        dpg.add_button(label="Frequencia de corte", callback=callback_botao_passabaixa)

    #------ 3.2.2 --- plotagem gráfico ------# 
    
    with dpg.plot(label="Sensores - Tensão (MPa)", height=-1, width=-1):
        dpg.add_plot_legend()
        xaxis = dpg.add_plot_axis(dpg.mvXAxis, label="Tempo (s)")
        yaxis = dpg.add_plot_axis(dpg.mvYAxis, label="Tensão (MPa)")
        
        if len(x_data) > 0:
            
            colunas_disponiveis = df_sensores.columns
            
            # Vamos plotar os primeiros 5 canais encontrados
            for i in range(min(18, len(colunas_disponiveis))):
                col_name = colunas_disponiveis[i]
                y_sensor = df_sensores[col_name].tolist()
                
                # Adiciona a linha ao gráfico
                # O label será "Canal 7", "Canal 8" baseado na lógica original (+7)
                dpg.add_line_series(x_data, y_sensor, parent=yaxis, label=f"Canal {7+i:02d}", tag=f"serie_canal_{i}")
                
            # Ajuste de Zoom automático
            dpg.fit_axis_data(yaxis)
            dpg.fit_axis_data(xaxis)
            
        else:
            dpg.add_text("Erro ao carregar dados.")

#dpg.bind_item_theme("Primary Window", white_color)

dpg.create_viewport(title='Analise Grafica', width=1000, height=600)
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.set_primary_window("Primary Window", True)
dpg.start_dearpygui()
dpg.destroy_context()