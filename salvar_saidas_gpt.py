import os
import json

def salvar_resposta(saida, pasta="./data/resultado/historico-gpt-teste4-gpt4o-mini"):
    """Salva a resposta em um arquivo dentro da pasta especificada."""
    os.makedirs(pasta, exist_ok=True)  # Garante que a pasta exista
    num_arquivos = len(os.listdir(pasta)) + 1
    caminho_arquivo = os.path.join(pasta, f"resposta_{num_arquivos}.txt")
    
    with open(caminho_arquivo, "w", encoding="utf-8") as f:
        json.dump(saida, f)
    
    print(f"Resposta salva em: {caminho_arquivo}")