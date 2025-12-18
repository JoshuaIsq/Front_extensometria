import dearpygui.dearpygui as dpg

dpg.create_context()

# --- LÓGICA DE EDIÇÃO "INLINE" (Clicar para Editar) ---

def alternar_para_edicao(sender, app_data, user_data):
    """
    Ao clicar no texto (botão), esconde ele e mostra a caixa de edição.
    """
    tag_botao, tag_input = user_data
    
    # 1. Esconde o botão de visualização
    dpg.hide_item(tag_botao)
    
    # 2. Mostra a caixa de texto
    dpg.show_item(tag_input)
    
    # 3. Dá foco automático (para digitar direto)
    dpg.focus_item(tag_input)

def confirmar_edicao(sender, app_data, user_data):
    """
    Ao apertar ENTER, salva o valor e volta a mostrar como texto.
    """
    tag_botao, tag_input, callback_mudanca, tag_alvo = user_data
    
    # 1. Pega o novo texto digitado
    novo_texto = dpg.get_value(sender)
    
    # 2. Atualiza o rótulo do botão (para exibir o novo nome)
    dpg.configure_item(tag_botao, label=novo_texto)
    
    # 3. Chama a função que realmente altera a janela/gráfico
    if callback_mudanca:
        callback_mudanca(tag_alvo, novo_texto)
    
    # 4. Troca de volta (Esconde Input, Mostra Botão)
    dpg.hide_item(tag_input)
    dpg.show_item(tag_botao)

def criar_texto_editavel(label_descritivo, valor_inicial, callback_mudanca, tag_alvo):
    """
    Cria um componente visual que parece texto, mas vira input ao clicar.
    """
    with dpg.group(horizontal=True):
        # Texto fixo descritivo (ex: "Título do Eixo:")
        dpg.add_text(f"{label_descritivo}: ")
        
        # O truque: Criamos o botão e o input ANTES, mas deixamos um invisível
        
        # 1. O INPUT (Invisível inicialmente)
        tag_input = dpg.add_input_text(default_value=valor_inicial, width=250, show=False, on_enter=True)
        
        # 2. O BOTÃO (Visível, small=True para parecer texto clicável)
        tag_botao = dpg.add_button(label=valor_inicial, small=True)
        
        # Configura os callbacks cruzados (um precisa saber o ID do outro)
        dpg.configure_item(tag_botao, callback=alternar_para_edicao, user_data=[tag_botao, tag_input])
        dpg.configure_item(tag_input, callback=confirmar_edicao, user_data=[tag_botao, tag_input, callback_mudanca, tag_alvo])

# --- Função que aplica a mudança no DPG ---

def aplicar_mudanca_no_item(tag_item, novo_texto):
    # Verifica se o item existe antes de tentar mudar
    if dpg.does_item_exist(tag_item):
        dpg.configure_item(tag_item, label=novo_texto)

# --- Interface Principal ---

with dpg.window(tag="minha_janela_principal", label="Janela Principal", width=600, height=600):
    
    dpg.add_text("EDITAR NOMES (Clique nos valores destacados para alterar):", color=(0, 255, 0))
    dpg.add_separator()
    dpg.add_spacing(count=5)

    # --- LISTA DE PROPRIEDADES EDITÁVEIS ---
    # Aqui criamos os campos mágicos. 
    # Passamos: Nome Descritivo, Valor Inicial, Função de Update, Tag do Item Real
    
    criar_texto_editavel("Título da Janela", "Janela Principal", aplicar_mudanca_no_item, "minha_janela_principal")
    criar_texto_editavel("Título do Gráfico", "Gráfico de Sensores", aplicar_mudanca_no_item, "meu_grafico")
    criar_texto_editavel("Eixo X", "Tempo (s)", aplicar_mudanca_no_item, "meu_eixo_x")
    criar_texto_editavel("Eixo Y", "Deformação (uE)", aplicar_mudanca_no_item, "meu_eixo_y")
    criar_texto_editavel("Legenda da Linha", "Dados Brutos", aplicar_mudanca_no_item, "minha_linha_dados")

    dpg.add_spacing(count=5)
    dpg.add_separator()

    # --- O GRÁFICO ---
    with dpg.plot(tag="meu_grafico", label="Gráfico de Sensores", height=-1, width=-1):
        dpg.add_plot_legend()
        
        dpg.add_plot_axis(dpg.mvXAxis, label="Tempo (s)", tag="meu_eixo_x")
        
        with dpg.plot_axis(dpg.mvYAxis, label="Deformação (uE)", tag="meu_eixo_y"):
            # Dados de exemplo
            dpg.add_line_series([0, 1, 2, 3, 4], [10, 15, 13, 17, 40], label="Dados Brutos", tag="minha_linha_dados")

dpg.create_viewport(title='Exemplo Edicao Inline', width=620, height=650)
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()