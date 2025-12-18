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

"""Abre arquivo de medição para análise experimental de tensões

        O método abre o arquivo de medição podendo esse no formato
        .txt. É criado o Dataframe da medição e o
        vetor contendo os registros temporais de todas as entradas."""

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
                
            #1.2 ------ Criando timestamp das horas -------- #Organizar melhor aqui depois
            time_cols = txt_file.iloc[:, 0:6] 
            time_cols.columns = ["day", "month", "year", "hour", "minute", "second"]
            timestamp = pd.to_datetime(time_cols)
            df_temp = pd.DataFrame({'timestamp': timestamp})
            df_data = pd.concat([df_temp, txt_file.iloc[:, 7:]], axis=1)
            df_sorted = df_data.sort_values(by='timestamp').reset_index(drop=True)
            start_time = df_sorted['timestamp'].iloc[0]
            #QUALQUER COISA APAGUE ESSE DEF
            def select_axes(type):
                if type == 'seconds':
                    eixo_x_segundos = ((df_sorted['timestamp'] - start_time).dt.total_seconds()).tolist()
                    return eixo_x_segundos
                elif type == 'hours':
                    eixo_x_segundos = ((df_sorted['timestamp'] - start_time).dt.total_seconds/3600()).tolist()
                    return eixo_x_segundos
                elif type == 'days':
                    eixo_x_segundos = ((df_sorted['timestamp'] - start_time).dt.total_seconds/86400()).tolist()
                    return eixo_x_segundos
                else:
                    eixo_x_segundos = ((df_sorted['timestamp'] - start_time).dt.total_seconds()).tolist()
                    return  eixo_x_segundos
            eixo_x = select_axes(type)
            sensores_df = df_sorted.iloc[:, 1:].fillna(0) * calibration 
            sensores_df = sensores_df.interpolate(method='linear', limit_direction='both').fillna(0)
            sensores_df.columns = [str(c) for c in sensores_df.columns]
            #alterando nomes dos sensores
            amount_data = len(sensores_df.columns)
            new_name = [str(i + 1) for i in range(amount_data)]
            sensores_df.columns = new_name

            print(f"Sucesso! {len(eixo_x)} pontos, {len(sensores_df.columns)} canais.")
            return eixo_x, sensores_df
            
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
    
#2.6 ----- Remover outliers ---- #
def remove_outliers(df, window, thresh=3, verbose=False):
    df_copia = df.copy()
    outlier_mask = indentify_outliers(df_copia, window, thresh, verbose)
    df_copia = df_copia.mask(outlier_mask)
    df_copia = df_copia.interpolate(method='linear', limit_direction='both').fillna(0)

    return df_copia.round(4)



# 3. ---------- Criação de botões ---------- #
""""São chamadas as funções de processamento e plotagem dos dados
e são aplicadas a ponteiros na interface gráfica, para em seguida 
atualizar o gráfico conforme os filtros são aplicados."""

# 3.1 --------- calculando taxa de plotagem ----- #

def processar_e_plotar(sender, app_data, user_data):

    if df_sensores.empty: #Se estiver sem nada importado, so fica vazio
        return
    
    df_trabalho = df_sensores.copy()
    
    if len(x_data) > 1:
        tempo_total = x_data[-1] - x_data[0]

        if tempo_total > 0:
            taxa_real = len(x_data) / tempo_total
        else:
            taxa_real = 1.0 
    else:
        taxa_real = 1.0

    print(f"Processando... Taxa: {taxa_real:.2f} Hz")

    # --- 3.2 Adicionando filtros um após o outro ---
    
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
    
    option_date = load_data_converte.select_axes(type)
    select = dpg.add_combo(items=option_date, label="Selecione medida de tempo", default_value=option_date[0], label="Tipo de eixo X")
    # 3.3 ------ PLOTAGEM ---------
    dpg.delete_item("eixo_y", children_only=True)


    for col_name in colunas_disponiveis:
        tag_check = checkbox_tags.get(col_name)
        if tag_check and dpg.get_value(tag_check):
            if col_name in df_trabalho.columns:
                value_y = df_trabalho[col_name].tolist()
                dpg.add_line_series(x_data, value_y, parent="eixo_y", label=f"Sensor {col_name}")
    


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
            dpg.add_checkbox(label=f"Sensor {col}", tag=tag_chk, default_value=estado, callback=processar_e_plotar, parent="grupo_lista_canais")
        
        # 3. Plota
        processar_e_plotar(None, None, None)
        # Ajusta o zoom para os novos dados
        dpg.fit_axis_data("eixo_x")
        dpg.fit_axis_data("eixo_y")



#4. --------------- Interface -------------------- #
dpg.create_context()

#4.1 --------- Cor das janelas (alteração) --------#

with dpg.theme() as tema_claro:
    #o mvall passa o fundo branco para todo mundo
    with dpg.theme_component(dpg.mvAll):
        dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (255, 255, 255, 255))
        dpg.add_theme_color(dpg.mvThemeCol_ChildBg, (255, 255, 255, 255))
        dpg.add_theme_color(dpg.mvThemeCol_PopupBg, (240, 240, 240, 240))
        dpg.add_theme_color(dpg.mvThemeCol_Text, (0, 0, 0))
        dpg.add_theme_color(dpg.mvThemeCol_Button, (240, 240, 240))
        dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (240, 240, 240))
        dpg.add_theme_color(dpg.mvThemeCol_Border, (240, 240, 240))

#------------------------------------------------



#4.1.1 Cria o seletor de arquivos 
with dpg.file_dialog(directory_selector=False, show=False, callback=select_archive, tag="file_dialog_id", width=700, height=400):
    dpg.add_file_extension(".txt", color=(0, 255, 0, 255))
    dpg.add_file_extension(".*")

#4.2 ----- Janelas principais ------#

with dpg.window(tag="Primary Window"):
    dpg.add_text("VISUALIZADOR DE EXTENSOMETRIA", color=(0, 0, 0))
    dpg.add_spacer(width=50)
    dpg.add_button(label="Selecionar aquivo: ", callback=lambda: dpg.show_item("file_dialog_id"))

    #4.2.1 ---- Botões -----

    with dpg.group(horizontal=True):

        with dpg.group(horizontal=True):
            # Onde você digita o valor da média
            dpg.add_input_int(default_value=0, width=90, tag="input_janela_mm")
            # O botão que chama a função que criamos acima
            dpg.add_button(label="Aplicar Média Móvel", callback=processar_e_plotar, )


        dpg.add_spacer(width=20)

        #dpg.add_text("Controle de Offset (Zerar):")
        
        with dpg.group(horizontal=True):
            dpg.add_input_int(default_value=0, width=90, tag="input_offset", min_value=0)
            dpg.add_button(label="Aplicar Offset", callback=processar_e_plotar)

        dpg.add_spacer(width=20)

        #dpg.add_text("Frequecia de corte passa baixa")

        with dpg.group(horizontal=True):
            dpg.add_input_float(default_value=0.00, width=90, tag='input_passabaixa', min_value=0.00)
            #dpg.add_input_int(default_value=0, width=90, tag="input_order", min_value=1, label="Ordem")
            dpg.add_button(label="Aplicar passa baixa", callback=processar_e_plotar)
            #dpg.add_button(label="Adicionar Ordem", callback=processar_e_plotar)

        dpg.add_spacer(width=20)


        with dpg.group(horizontal=True):
            dpg.add_input_float(default_value=0.00, width=90, tag="input_highpass", min_value=0.00)
            dpg.add_button(label="Aplicar passa alta", callback=processar_e_plotar)

        with dpg.group(horizontal=True):
            dpg.add_input_int(default_value=0, width=90, tag="input_outliers", min_value=0)
            dpg.add_button(label="Remover Outliers", callback=processar_e_plotar)

        #dpg.add_separator()

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


dpg.bind_item_theme("Primary Window", tema_claro)


#----- Exibição ---------#

processar_e_plotar(None, None, None)
dpg.create_viewport(title='Analise Grafica', width=1000, height=600)
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.set_primary_window("Primary Window", True)
dpg.start_dearpygui()
dpg.destroy_context()