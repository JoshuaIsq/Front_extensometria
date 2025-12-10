import pandas as pd
import numpy as np
import dearpygui.dearpygui as dpg
import scipy as sp
from scipy import signal

x_data = []
df_sensores = pd.DataFrame()
checkbox_tags = {} 
colunas_disponiveis = []

#1 ---------------- Carregar dados do arquivo ------------------ #
def load_data_converte(filename, calibration):

    passo = 1
    if filename.endswith(".txt"):

        try:
            #1.1 ------ Importando o meu arquivo txt com leitura extensometrica --------

            txt_file = pd.read_csv(filename, sep=r'[;\s]+', header=None, engine="python", on_bad_lines="skip") 

            #1.2 ---- Transformando possiveis linhas problematicas em linhas funcionais -----

            # Vamos garantir que as primeiras 6 colunas (tempo) sejam NÚMEROS
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
            sensores_df = df_sorted.iloc[:, 1:].fillna(0) * calibration
            #Cria uma interpolação caso tenham buracos no gráfico
            sensores_df = sensores_df.interpolate(method='linear', limit_direction='both').fillna(0)

            # Renomeia colunas para ficar bonito (Canal 1, Canal 2...) se forem números puros
            sensores_df.columns = [str(c) for c in sensores_df.columns]

            print(f"Sucesso! {len(eixo_x_segundos)} pontos, {len(sensores_df.columns)} canais.")
            return eixo_x_segundos, sensores_df
            
        except Exception as e:
            print(f"Erro ao carregar: {e}")
            return [], pd.DataFrame()
        

    
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

#2.4 ------- Filtro Passa Alta ------- #

def filter_high_pass(df, freq_corte, freq_rate, order):
    df_copia = df.copy()
    nyquisfreq = 0.5 * freq_rate
    filter_high_pass = freq_corte/nyquisfreq
    b, a = signal.butter(order, filter_high_pass, btype="high")
    for col in df_copia.columns:
        df_copia[col] = signal.filtfilt(b, a, df_copia[col])
    return df_copia.round(4)



# 3. ---------- Criação de botões ---------- #

# 3.1 --------- calculando taxa de plotagem ----- #

def processar_e_plotar(sender, app_data, user_data):

    if df_sensores.empty: #Se estiver sem nada importado, so fica vazio
        return
    
    df_trabalho = df_sensores.copy()
    
    if len(x_data) > 1:
        # Pega o tempo total (Fim - Início)
        tempo_total = x_data[-1] - x_data[0]

        # calcula a média
        if tempo_total > 0:
            # Taxa = Quantos pontos existem / Quanto tempo durou
            taxa_real = len(x_data) / tempo_total
        else:
            taxa_real = 1.0 # Evita divisão por zero se o arquivo tiver tempo zerado
    else:
        taxa_real = 1.0

    print(f"Processando... Taxa: {taxa_real:.2f} Hz")

    # --- 3.2 Adicionando filtros um após o outro ---
    
    # Passo A: Offset
    n_offset = dpg.get_value("input_offset")
    if n_offset > 0:
        df_trabalho = adjust_offset(df_trabalho, n_offset)

    # Passo B: Média Móvel
    janela = dpg.get_value("input_janela_mm")
    if janela > 1:
        df_trabalho = media_movel(df_trabalho, janela)

    # Passo C: Passa Baixa
    corte_low = dpg.get_value("input_passabaixa")
    if corte_low > 0 and corte_low < (taxa_real / 2):
        df_trabalho = filter_low_pass(df_trabalho, corte_low, taxa_real, order=2)

    # Passo D: Passa Alta
    corte_high = dpg.get_value("input_highpass")
    if corte_high > 0 and corte_high < (taxa_real / 2):
        df_trabalho = filter_high_pass(df_trabalho, corte_high, freq_rate=taxa_real, order=2)

    # 3.3 ------ PLOTAGEM ---------
    dpg.delete_item("eixo_y", children_only=True)


    for col_name in colunas_disponiveis:
        tag_check = checkbox_tags.get(col_name)
        if tag_check and dpg.get_value(tag_check):
            if col_name in df_trabalho.columns:
                value_y = df_trabalho[col_name].tolist()
                dpg.add_line_series(x_data, value_y, parent="eixo_y", label=f"Canal {col_name}")
    
    
    '''for i in range(min(18, len(colunas))):
        col_name = colunas[i]
        y_vals = df_trabalho[col_name].tolist()
        dpg.add_line_series(x_data, y_vals, parent="Eixo Y", label=f"Canal {col_name}")'''
    #plotagem das colunas sem checkbox (apagar dps)

    # 3.4 ------- Criando o zoom ------ #
def callback_zomm(sender, app_data):
    x_min, x_max = app_data[0], app_data[1]
    y_min, y_max = app_data[2], app_data[3]
    dpg.set_axis_limits("eixo_X", x_min, x_max)
    dpg.set_axis_limits("eixo_y", y_min, y_max)

# --------- Seleção de arquivo ----------- #

def select_archive(sender, app_data):
    global x_data, df_sensores, colunas_disponiveis
    
    # app_data['file_path_name'] contém o caminho completo do arquivo escolhido
    caminho = app_data['file_path_name']
    
    # 1. Carrega os novos dados
    x_data, df_sensores = load_data_converte(caminho, 0.00003375)
    
    if len(x_data) > 0:
        colunas_disponiveis = df_sensores.columns.tolist()
        
        # 2. Reconstrói a lista de Checkboxes
        dpg.delete_item("grupo_lista_canais", children_only=True)
        checkbox_tags.clear()
        
        for col in colunas_disponiveis:
            tag_chk = f"chk_{col}"
            checkbox_tags[col] = tag_chk
            # Marca os 3 primeiros por padrão
            estado = True if col in colunas_disponiveis[:3] else False
            # Adiciona ao grupo que limpamos acima
            dpg.add_checkbox(label=f"Canal {col}", tag=tag_chk, default_value=estado, callback=processar_e_plotar, parent="grupo_lista_canais")
        
        # 3. Plota
        processar_e_plotar(None, None, None)
        # Ajusta o zoom para os novos dados
        dpg.fit_axis_data("eixo_x")
        dpg.fit_axis_data("eixo_y")



#4. --------------- Interface -------------------- #
dpg.create_context()

#4.1 --------- Cor das janelas --------#

with dpg.theme() as white_color:
    with dpg.theme_component(dpg.mvWindowAppItem):
        dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (255, 255, 255, 255))

# Cria o seletor de arquivos 
with dpg.file_dialog(directory_selector=False, show=False, callback=select_archive, tag="file_dialog_id", width=700, height=400):
    dpg.add_file_extension(".txt", color=(0, 255, 0, 255))
    dpg.add_file_extension(".*")

#4.2 ----- Janelas principais ------#

with dpg.window(tag="Primary Window"):
    dpg.add_text("VISUALIZADOR DE EXTENSOMETRIA", color=(0, 255, 255))
    dpg.add_spacer(width=50)
    dpg.add_button(label="Selecionar aquivo: ", callback=lambda: dpg.show_item("file_dialog_id"))

    #4.2.1 ---- Botões -----

    with dpg.group(horizontal=True):
        # Onde você digita o valor da média
        dpg.add_input_int(default_value=10, width=150, tag="input_janela_mm")
        # O botão que chama a função que criamos acima
        dpg.add_button(label="Aplicar Média Móvel", callback=processar_e_plotar)


    dpg.add_separator()

    dpg.add_text("Controle de Offset (Zerar):")
    
    with dpg.group(horizontal=True):
        dpg.add_input_int(default_value=10, width=150, tag="input_offset", min_value=1)
        dpg.add_button(label="Aplicar Offset", callback=processar_e_plotar)

    dpg.add_separator()

    dpg.add_text("Frequecia de corte passa baixa")

    with dpg.group(horizontal=True):
        dpg.add_input_float(default_value=10, width=150, tag='input_passabaixa', min_value=0.01)
        dpg.add_button(label="Frequencia de corte passa baixa", callback=processar_e_plotar)

    dpg.add_separator()


    with dpg.group(horizontal=True):
        dpg.add_input_float(default_value=10, width=150, tag="input_highpass", min_value=0.01)
        dpg.add_button(label="Frequencia corte passa alta", callback=processar_e_plotar)

    dpg.add_separator()

# 4.3 --- plotagem gráfico ------# 
    
#----- 4.3.1  Cria a "Prateleira" (Grupo Horizontal)
    with dpg.group(horizontal=True):
        
        # ---- 4.3.2 Cria a Caixa da Esquerda (Lista de Canais)
        with dpg.child_window(width=200, height=-1):
            dpg.add_text("Canais Disponíveis:")
            
            # Botão Auxiliar
            def toggle_all(sender, app_data):
                for col in colunas_disponiveis:
                    dpg.set_value(checkbox_tags[col], True)
                processar_e_plotar(None, None, None)
            
            dpg.add_button(label="Marcar Todos", callback=toggle_all)
            dpg.add_separator()

            with dpg.group(tag="grupo_lista_canais"):

            # Cria os Checkboxes
                for col in colunas_disponiveis:
                    tag_chk = f"chk_{col}"
                    checkbox_tags[col] = tag_chk
                    
                    # Marcano os 3 primeiros canais
                    comeca_marcado = True if col in colunas_disponiveis[:3] else False
                    
                    # Cria o checkbox e avisa que se clicar, chama o 'processar_e_plotar'
                    dpg.add_checkbox(label=f"Canal {col}", tag=tag_chk, default_value=comeca_marcado, callback=processar_e_plotar)

        # 4.3.3 Cria a Caixa da Direita (O Gráfico)
        with dpg.plot(label="Analise", height=-1, width=-1, query=True, callback=callback_zomm):
            dpg.add_plot_legend()
            
            xaxis = dpg.add_plot_axis(dpg.mvXAxis, label="Tempo (s)", tag="eixo_x")
            yaxis = dpg.add_plot_axis(dpg.mvYAxis, label="Tensão (MPa)", tag="eixo_y")


#dpg.bind_item_theme("Primary Window", white_color)

#----- Exibição ---------#

processar_e_plotar(None, None, None)
dpg.create_viewport(title='Analise Grafica', width=1000, height=600)
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.set_primary_window("Primary Window", True)
dpg.start_dearpygui()
dpg.destroy_context()