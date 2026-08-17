import whisper

# 1. Carrega o modelo do Whisper
# Opções de modelo: 'tiny', 'base', 'small', 'medium', 'large'
print("Carregando o modelo...")
model = whisper.load_model("base")

# 2. Faz a transcrição do arquivo de áudio
nome_do_arquivo = "seu_audio.mp3" # Substitua pelo nome do seu arquivo
print("Transcrevendo...")
result = model.transcribe(nome_do_arquivo)

# 3. Salva o texto transcrito em um arquivo .txt
caminho_txt = "transcricao.txt"
with open(caminho_txt, "w", encoding="utf-8") as file:
    file.write(result["text"])

print(f"Transcrição concluída com sucesso! Salva em '{caminho_txt}'.")