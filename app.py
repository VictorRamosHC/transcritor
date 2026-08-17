import os
from flask import Flask, render_template, request, jsonify
import whisper

app = Flask(__name__)

# Carrega o modelo ao iniciar (otimizado para serverless)
print("Carregando modelo do Whisper...")
model = whisper.load_model("base")

UPLOAD_FOLDER = "/tmp/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/transcrever", methods=["POST"])
def transcrever():
    if "arquivo" not in request.files:
        return jsonify({"erro": "Nenhum arquivo enviado"}), 400
    
    file = request.files["arquivo"]
    if file.filename == "":
        return jsonify({"erro": "Nenhum arquivo selecionado"}), 400

    # Salva o arquivo enviado temporariamente
    caminho_arquivo = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(caminho_arquivo)

    try:
        # Transcreve o MP4 / áudio
        result = model.transcribe(caminho_arquivo)
        
        # Remove o arquivo do disco após transcrever para não acumular
        os.remove(caminho_arquivo)

        return jsonify({"texto": result["text"]})
    except Exception as e:
        if os.path.exists(caminho_arquivo):
            os.remove(caminho_arquivo)
        return jsonify({"erro": str(e)}), 500

# Vercel precisa que o app seja exportável
if __name__ == "__main__":
    app.run(debug=True, port=5000)