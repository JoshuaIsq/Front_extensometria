import pandas as pd
import numpy as np
import dearpygui.dearpygui as dpg
from scipy import signal
import os
import matplotlib.pyplot as plt

# --- VARIÁVEIS GLOBAIS (Inicializadas vazias) ---
x_data = []
df_sensores = pd.DataFrame()
checkbox_tags = {} 
colunas_disponiveis = []

# 1. ---------------- Carregar e Limpar Dados ------------------ #
def load_data_converte(filepath, calibration=0.00003375):
    if filepath.endswith(".txt"):
        try:
            print(f"Lendo: {filepath} ...")
            # Leitura dos arquivos
            txt_file = pd.read_csv(filepath, sep=r'[;\s]+', header=None, engine="python", on_bad_lines='skip') 

            # Limpeza de Tempo
            cols_tempo_indices = [0, 1, 2, 3, 4, 5]
            for col in cols_tempo_indices:
                txt_file[col] = pd.to_numeric(txt_file[col], errors='coerce')
            
            txt_file = txt_file.dropna(subset=cols_tempo_indices)
            txt_file[cols_tempo_indices] = txt_file[cols_tempo_indices].astype(int)
                
            time_cols = txt_file.iloc[:, 0:6]
            time_cols.columns = ["day", "month", "year", "hour", "minute", "second"]
            timestamp = pd.to_datetime(time_cols)
            
            df_temp = pd.DataFrame({'timestamp': timestamp})
            # Pega da coluna 7 em diante (Sensores)
            df_data = pd.concat([df_temp, txt_file.iloc[:, 7:]], axis=1)
            
            df_sorted = df_data.sort_values(by='timestamp').reset_index(drop=True)
            
            start_time = df_sorted['timestamp'].iloc[0]
            eixo_x_segundos = (df_sorted['timestamp'] - start_time).dt.total_seconds().tolist()
            
            # Sensores
            sensores_df = df_sorted.iloc[:, 1:].fillna(0) * calibration
            sensores_df = sensores_df.where((sensores_df > -5000) & (sensores_df < 5000), np.nan)
            sensores_df = sensores_df.interpolate(method='linear', limit_direction='both').fillna(0)

            # Renomeia colunas para ficar bonito (Canal 1, Canal 2...) se forem números puros
            sensores_df.columns = [str(c) for c in sensores_df.columns]

            print(f"Sucesso! {len(eixo_x_segundos)} pontos, {len(sensores_df.columns)} canais.")
            return eixo_x_segundos, sensores_df
            
        except Exception as e:
            print(f"Erro ao carregar: {e}")
            return [], pd.DataFrame()
    return [], pd.DataFrame()


# 2. ---------------- Funções Matemáticas ------------------ #
# (Mantidas iguais ao anterior)

def media_movel(df, janela):
    if janela <= 1: return df
    df_copia = df.copy()
    return df_copia.rolling(window=int(janela), min_periods=1, center=True).mean()

def adjust_offset(df, n_linhas):
    if n_linhas <= 0: return df
    df_copia = df.copy()
    adjust = df_copia.iloc[:int(n_linhas)].mean()
    return df_copia - adjust

def filter_low_pass(df, cut_freq, sample_rate, order=2):
    if cut_freq <= 0: return df
    df_copia = df.copy()
    nyquisfreq = 0.5 * sample_rate
    if cut_freq >= nyquisfreq: return df
    b, a = signal.butter(order, cut_freq / nyquisfreq, btype="low")
    for col in df_copia.columns:
        df_copia[col] = signal.filtfilt(b, a, df_copia[col])
    return df_copia

def filter_high_pass(df, freq_corte, freq_rate, order=2):
    if freq_corte <= 0: return df
    df_copia = df.copy()
    nyquisfreq = 0.5 * freq_rate
    if freq_corte >= nyquisfreq: return df
    b, a = signal.butter(order, freq_corte / nyquisfreq, btype="high")
    for col in df_copia.columns:
        df_copia[col] = signal.filtfilt(b, a, df_copia[col])
    return df_copia


# 3. ---------- Lógica Central (Pipeline) ---------- #

def processar_e_plotar(sender, app_data, user_data):
    # Se não tiver dados carregados, não faz nada
    if df_sensores.empty:
        return

    df_trabalho = df_sensores.copy()
    
    # Cálculo Taxa Real
    if len(x_data) > 1:
        tempo_total = x_data[-1] - x_data[0]
        taxa_real = len(x_data) / tempo_total if tempo_total > 0 else 1.0
    else:
        taxa_real = 1.0

    print(f"Processando... Taxa: {taxa_real:.2f} Hz")

    # Filtros
    n_offset = dpg.get_value("input_offset")
    if n_offset > 0: df_trabalho = adjust_offset(df_trabalho, n_offset)

    janela = dpg.get_value("input_janela_mm")
    if janela > 1: df_trabalho = media_movel(df_trabalho, janela)

    corte_low = dpg.get_value("input_passabaixa")
    if corte_low > 0: df_trabalho = filter_low_pass(df_trabalho, corte_low, taxa_real)

    corte_high = dpg.get_value("input_highpass")
    if corte_high > 0: df_trabalho = filter_high_pass(df_trabalho, corte_high, taxa_real)

    # Plotagem
    dpg.delete_item("eixo_y", children_only=True)
    
    # Usa a lista global de colunas disponíveis
    for col_name in colunas_disponiveis:
        tag_chk = checkbox_tags.get(col_name)
        # Só desenha se o checkbox existe e está marcado
        if tag_chk and dpg.get_value(tag_chk):
            if col_name in df_trabalho.columns:
                y_vals = df_trabalho[col_name].tolist()
                dpg.add_line_series(x_data, y_vals, parent="eixo_y", label=f"Canal {col_name}")


def callback_zoom(sender, app_data):
    dpg.set_axis_limits("eixo_x", app_data[0], app_data[1])
    dpg.set_axis_limits("eixo_y", app_data[2], app_data[3])


# --- 4. SELEÇÃO DE ARQUIVOS (NOVO) ---

def ao_escolher_arquivo(sender, app_data):
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

# --- NOVO: FUNÇÃO DE EXPORTAÇÃO PARA PNG (MATPLOTLIB) ---
def exportar_grafico(sender, app_data, user_data):
    if df_visualizacao.empty or len(x_data) == 0:
        print("Nada para exportar!")
        return

    print("Gerando imagem... (Pode levar alguns segundos)")
    
    # 1. Cria uma figura invisível do Matplotlib
    plt.figure(figsize=(12, 6), dpi=150) # Tamanho e Resolução
    
    # 2. Plota os mesmos canais que estão marcados no Dear PyGui
    for col_name in colunas_disponiveis:
        tag_chk = checkbox_tags.get(col_name)
        if tag_chk and dpg.get_value(tag_chk):
            if col_name in df_visualizacao.columns:
                # Plota a linha
                plt.plot(x_data, df_visualizacao[col_name], label=f"Canal {col_name}", linewidth=1)

    # 3. Configura o visual "Relatório" (Fundo Branco)
    plt.title("Relatório de Extensometria")
    plt.xlabel("Tempo (s)")
    plt.ylabel("Tensão (MPa)")
    plt.grid(True, linestyle='--', alpha=0.7) # Grade pontilhada
    plt.legend() # Legenda automática
    plt.tight_layout()

    # 4. Salva o arquivo
    nome_arquivo = "Grafico_Exportado.png"
    plt.savefig(nome_arquivo)
    plt.close() # Limpa a memória
    
    print(f"GRÁFICO SALVO COM SUCESSO: {nome_arquivo}")


# 5. ---------------- Interface Gráfica ------------------ #
dpg.create_context()

with dpg.theme() as white_color:
    with dpg.theme_component(dpg.mvWindowAppItem):
        dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (255, 255, 255, 255))

# Cria o seletor de arquivos (invisível até ser chamado)
with dpg.file_dialog(directory_selector=False, show=False, callback=ao_escolher_arquivo, tag="file_dialog_id", width=700, height=400):
    dpg.add_file_extension(".txt", color=(0, 255, 0, 255))
    dpg.add_file_extension(".*")

with dpg.window(tag="Primary Window"):
    
    # --- CABEÇALHO ---
    with dpg.group(horizontal=True):
        dpg.add_text("VISUALIZADOR DE EXTENSOMETRIA", color=(0, 255, 255))
        dpg.add_spacer(width=50)
        # Botão para abrir o seletor
        dpg.add_button(label="ABRIR ARQUIVO", callback=lambda: dpg.show_item("file_dialog_id"))

    dpg.add_separator()

    # --- FILTROS ---
    with dpg.group(horizontal=True):
        dpg.add_text("Média Móvel:")
        dpg.add_input_int(default_value=0, width=100, tag="input_janela_mm", min_value=0)
        dpg.add_button(label="Aplicar", callback=processar_e_plotar)

    with dpg.group(horizontal=True):
        dpg.add_text("Offset:")
        dpg.add_input_int(default_value=0, width=100, tag="input_offset", min_value=0)
        dpg.add_button(label="Aplicar", callback=processar_e_plotar)

    with dpg.group(horizontal=True):
        dpg.add_text("Low Pass (Hz):")
        dpg.add_input_float(default_value=0.0, width=100, tag='input_passabaixa', min_value=0.0)
        dpg.add_button(label="Aplicar", callback=processar_e_plotar)

    with dpg.group(horizontal=True):
        dpg.add_text("High Pass (Hz):")
        dpg.add_input_float(default_value=0.0, width=100, tag="input_highpass", min_value=0.0)
        dpg.add_button(label="Aplicar", callback=processar_e_plotar)

    dpg.add_separator()
    
    # --- ÁREA PRINCIPAL ---
    with dpg.group(horizontal=True):
        
        # COLUNA ESQUERDA
        with dpg.child_window(width=200, height=-1):
            dpg.add_text("Canais Disponíveis:")
            
            def toggle_all(sender, app_data):
                for col in colunas_disponiveis:
                    dpg.set_value(checkbox_tags[col], True)
                processar_e_plotar(None, None, None)
            dpg.add_button(label="Marcar Todos", callback=toggle_all)
            dpg.add_separator()

            # Grupo vazio onde os checkboxes serão criados dinamicamente
            dpg.add_group(tag="grupo_lista_canais")

        # COLUNA DIREITA
        with dpg.plot(label="Analise", height=-1, width=-1, query=True, callback=callback_zoom):
            dpg.add_plot_legend()
            xaxis = dpg.add_plot_axis(dpg.mvXAxis, label="Tempo (s)", tag="eixo_x")
            yaxis = dpg.add_plot_axis(dpg.mvYAxis, label="Tensão (MPa)", tag="eixo_y")

# Inicialização (Sem carregar dados, espera o usuário)
dpg.create_viewport(title='Analise Grafica', width=1200, height=800)
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.set_primary_window("Primary Window", True)
dpg.start_dearpygui()
dpg.destroy_context()
