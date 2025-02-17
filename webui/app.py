from flask import Flask, request, render_template, send_file
from transformers import pipeline
from melo.api import TTS
import os

app = Flask(__name__)

def process(input_audio_fp):
    pipe = pipeline(model="justanotherinternetguy/whisper-small-sep28")
    text = pipe(input_audio_fp)["text"]
    
    device = 'cuda'  # Will automatically use GPU if available
    model = TTS(language='EN', device=device)
    speaker_ids = model.hps.data.spk2id

    output_path = 'output/en-us.wav'  # Ensure this directory exists
    model.tts_to_file(text, speaker_ids['EN-Default'], output_path, speed=1)
    return text, output_path

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'audio_file' not in request.files:
            return "No file part", 400
        
        file = request.files['audio_file']
        if file.filename == '':
            return "No selected file", 400
        
        input_audio_fp = f'uploads/{file.filename}'
        file.save(input_audio_fp)

        text, output_path = process(input_audio_fp)

        return render_template('result.html', text=text, audio_file=output_path)

    return render_template('index.html')

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_file(f'output/{filename}', as_attachment=True)

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('output', exist_ok=True)
    app.run(debug=True)
