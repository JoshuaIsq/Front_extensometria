from chamadas import processar_e_plotar, callback_zomm, select_archive, open_tendency
from Import_And_Math import DataStorage
from Import_And_Math import actual_tendency
import dearpygui.dearpygui as dpg



"""Interface gráfica para visualização e análise de dados de extensometria.
    Utiliza Dear PyGui para criar a interface do usuário, permitindo a seleção de arquivos,
    aplicação de filtros e ajustes, e plotagem dos dados processados."""

dpg.create_context()

with dpg.font_registry():
    default_font = dpg.add_font("C:\\Windows\\Fonts\\Times.ttf", 20)

dpg.bind_font(default_font)

#4.1 --------- Cor das janelas  --------#

with dpg.theme() as tema_claro:
    with dpg.theme_component(dpg.mvAll):
        dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (255, 255, 255, 255))
        dpg.add_theme_color(dpg.mvThemeCol_ChildBg, (255, 255, 255, 255))
        dpg.add_theme_color(dpg.mvThemeCol_PopupBg, (255, 255, 255, 255))
        dpg.add_theme_color(dpg.mvThemeCol_Text, (0, 0, 0))
        dpg.add_theme_color(dpg.mvThemeCol_Button, (240, 240, 240))
        dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (255, 255, 255))
        dpg.add_theme_color(dpg.mvThemeCol_Border, (255, 255, 255))

#---------- Seletor de arquivos ------------
with dpg.file_dialog(directory_selector=False, show=False, callback=select_archive, tag="file_dialog_id", width=700, height=400):
    dpg.add_file_extension(".txt", color=(0, 255, 0, 255))
    dpg.add_file_extension(".*")

# ----- Janela principal e botões ------#

with dpg.window(tag="Primary Window"):
    dpg.add_text("VISUALIZADOR DE EXTENSOMETRIA", color=(0, 0, 0),)
    dpg.add_spacer(width=50)
    dpg.add_button(label="Selecionar aquivo: ", callback=lambda: dpg.show_item("file_dialog_id"))

    # ---- Botões -----

    with dpg.group(horizontal=True):


        with dpg.group(horizontal=True):
            with dpg.group(horizontal=False):
                dpg.add_text("Janela Média Móvel:")
                dpg.add_input_int(default_value=0, width=90, tag="input_janela_mm")
                dpg.add_spacer(height=20)
                dpg.add_button(label="Aplicar Média Móvel", callback=processar_e_plotar, )

        dpg.add_spacer(width=20)

        with dpg.group(horizontal=True):
            with dpg.group(horizontal=False):
                dpg.add_text("Ajuste de Offset:")
                dpg.add_input_int(default_value=0, width=90, tag="input_offset", min_value=0)
                dpg.add_spacer(height=20)
                dpg.add_button(label="Aplicar Offset", callback=processar_e_plotar)

        dpg.add_spacer(width=20)

        with dpg.group(horizontal=True):
            with dpg.group(horizontal=False):
                dpg.add_text("Frequência de corte passa baixa:")
                dpg.add_input_float(default_value=0.00, width=90, tag='input_passabaixa', min_value=0.00, label="Freq. Corte (Hz)")
                dpg.add_input_int(default_value=2, width=90, tag="input_order_low", min_value=1, min_clamped=True, label="Ordem")
                dpg.add_button(label="Aplicar passa baixa e ordem", callback=processar_e_plotar)
                
        dpg.add_spacer(width=20)

        with dpg.group(horizontal=True):
            with dpg.group(horizontal=False):
                dpg.add_text("Frequência de corte passa alta:")
                dpg.add_input_float(default_value=0.00, width=90, tag='input_highpass', min_value=0.00, label="Freq. Corte (Hz)")
                dpg.add_input_int(default_value=2, width=90, tag="input_order_high", min_value=1, min_clamped=True, label="Ordem")  
                dpg.add_button(label="Aplicar passa alta e ordem", callback=processar_e_plotar)

        dpg.add_spacer(width=20)

        with dpg.group(horizontal=True):
            with dpg.group(horizontal=False):
                dpg.add_text("Remoção de Outliers:")
                dpg.add_input_int(default_value=0, width=90, tag="input_outliers", min_value=0)
                dpg.add_spacer(height=20)
                dpg.add_button(label="Remover Outliers", callback=processar_e_plotar)
        
        with dpg.group(horizontal=True):
            dpg.add_button(label="Ver tendencia", callback=open_tendency)

#  --- plotagem gráfico ------# 
    
#-----  Cria a "Prateleira" (Grupo Horizontal)
    with dpg.group(horizontal=True):
        
        # ---- 4.3.2 Cria a Caixa da Esquerda (Lista de Canais)
        with dpg.child_window(width=200, height=-1):
            dpg.add_text("Canais Disponíveis:")
            
            # Botão Auxiliar
            def toggle_all(sender, app_data):
                for col in DataStorage.colunas_disponiveis:
                    dpg.set_value(DataStorage.checkbox_tags[col], True)
                processar_e_plotar(None, None, None)
            
            dpg.add_button(label="Marcar Todos", callback=toggle_all)
            dpg.add_separator()

            with dpg.group(tag="grupo_lista_canais"):

            # Cria os Checkboxes
                for col in DataStorage.colunas_disponiveis:
                    tag_chk = f"chk_{col}"
                    DataStorage.checkbox_tags[col] = tag_chk
                    
                    # Marcano os 3 primeiros canais
                    comeca_marcado = True if col in DataStorage.colunas_disponiveis[:3] else False
                    
                    # Cria o checkbox e avisa que se clicar, chama o 'processar_e_plotar'
                    dpg.add_checkbox(label=f"Canal {col}", tag=tag_chk, default_value=comeca_marcado, callback=processar_e_plotar)

        # 4.3.3 Cria a Caixa da Direita (O Gráfico)
        with dpg.plot(label="Extensômetros superiores", height=-1, width=-1, query=True, callback=callback_zomm):
            dpg.add_plot_legend()
            
            xaxis = dpg.add_plot_axis(dpg.mvXAxis, label="Data / Hora", tag="eixo_x", time=True)
            yaxis = dpg.add_plot_axis(dpg.mvYAxis, label="Deslocamento (mm)", tag="eixo_y")


dpg.bind_item_theme("Primary Window", tema_claro)


#----- Exibição ---------#

processar_e_plotar(None, None, None)
dpg.create_viewport(title='Analise Grafica', width=1000, height=600)
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.set_primary_window("Primary Window", True)
dpg.start_dearpygui()
dpg.destroy_context()