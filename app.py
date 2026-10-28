import gradio as gr
from faster_whisper import WhisperModel
import tempfile
import os

# Carrega o modelo uma vez no startup do Space
print("Carregando modelo Whisper (base)...")
model = WhisperModel("base", device="cpu", compute_type="int8")
print("Modelo carregado.")


def transcrever(arquivo):
    if arquivo is None:
        return "Envie um arquivo de áudio ou vídeo primeiro."

    caminho = arquivo if isinstance(arquivo, str) else getattr(arquivo, "name", None)
    if not caminho or not os.path.exists(caminho):
        return "Arquivo inválido."

    try:
        segments, info = model.transcribe(
            caminho,
            beam_size=5,
            vad_filter=True,
            language=None,  # detecção automática
        )
        texto = "".join(seg.text for seg in segments).strip()
        if not texto:
            return "Nenhuma fala detectada no arquivo."
        return texto
    except Exception as e:
        return f"Erro ao transcrever: {e}"


with gr.Blocks(title="Transcrição com Whisper") as demo:
    gr.Markdown("# 🎙️ Transcrição com Whisper")
    gr.Markdown("Envie um vídeo MP4 ou arquivo de áudio e obtenha a transcrição completa em texto.")

    with gr.Row():
        entrada = gr.File(label="Arquivo de áudio/vídeo", file_types=["audio", "video"])
    btn = gr.Button("Transcrever", variant="primary")
    saida = gr.Textbox(label="Transcrição", lines=12, show_copy_button=True)

    btn.click(fn=transcrever, inputs=entrada, outputs=saida)
    entrada.change(fn=transcrever, inputs=entrada, outputs=saida)

    gr.Markdown("Formatos suportados: MP4, MP3, WAV, M4A e outros (via ffmpeg).")

if __name__ == "__main__":
    demo.launch()
