from flask import Flask, request, jsonify, json, send_file
import base64
from final import *
from flask_cors import CORS
from gtts import gTTS
from translate import Translator
from deep_translator import GoogleTranslator
import time
from flask_socketio import SocketIO
from io import BytesIO
import random

app = Flask(__name__)
socketio = SocketIO(app)

json_file_path = 'contexts.json'
load_dotenv()

with open(json_file_path, 'r') as json_file:
    chatbot_data = json.load(json_file)


#  A POST endpoint /set_target_language is defined for setting the target language for translation.
@app.route('/set_target_language', methods=['POST'])
def set_target_language():
    try:
        if request.headers['Content-Type'] == 'application/json':
            data = request.get_json()
            new_target_language = data.get('target_language')
            
            # Check if the provided language is valid (Hindi, English, or Tamil)
            valid_languages = ['hi', 'en', 'ta']
            if new_target_language in valid_languages:
                global target_language
                target_language = new_target_language
                return jsonify({"message": f"Target language set to {target_language}"}), 200
            else:
                return jsonify({"error": "Invalid target language. Please choose 'hi' for Hindi, 'en' for English, or 'ta' for Tamil."}), 400
        else:
            return jsonify({"error": "Invalid content type. Please use 'application/json'."}), 415
    except KeyError as ke:
        return jsonify({"error": f"KeyError: {str(ke)}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
    
# Pre LLM Translator
# Function to translate text using deep_translator
def transl_text(text, target_language='en'):
    try:
        translation = GoogleTranslator(source='auto', target=target_language).translate(text)
        return translation
    except Exception as e:
        print(f"Translation error: {e}")
        return None


# Function to translate text using deep_translator with retries and backoff
def translate_text_with_retries(text, target_language, max_retries=3):
    for retry_count in range(max_retries):
        try:
            translation = GoogleTranslator(source='en', target=target_language).translate(text)
            return translation
        except Exception as e:
            print(f"Translation request failed with error: {e}")
            if retry_count < max_retries - 1:
                wait_time = 2 ** retry_count  # Exponential backoff
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
    print("Maximum retries reached. Unable to translate text.")
    return None

# Function to split text into chunks
def chunk_text(text, max_chunk_length=500):
    chunks = [text[i:i + max_chunk_length] for i in range(0, len(text), max_chunk_length)]
    return chunks


# A POST endpoint /qa is defined for handling question-answering queries.
@app.route('/qa', methods=['POST'])
def qa_endpoint():
    try:
        print("running")
        if request.headers['Content-Type'] == 'application/json':
            data = request.get_json()
            question = data['question']
            english_question=transl_text(question)
            print(english_question)
            start_time = time.time()
            print(f"Start time: {start_time} seconds")

            
            # Call the retrieval-based QA function
            response = retrieval_qa(english_question)
            text_chunks = chunk_text(response)
            translated_text = ""
            for chunk in text_chunks:
             translated_text += translate_text_with_retries(chunk, target_language)
        
            # Print the translated text
            end_time = time.time()
            print(f"Total processing time: {end_time} seconds")
            
            #The response is changed to audio file
            tts = gTTS(text=translated_text , slow=False)
            audio_file_path = "converted.mp3"
            tts.save(audio_file_path)
            response=translated_text
            
            
            #making it a audio file path 
            with open(audio_file_path, 'rb' ) as audio_file:
                audio_data = audio_file.read()
                audio_base64 = base64.b64encode(audio_data).decode('utf-8')
            socketio.emit('qa_response', {'response': response, 'audio_output': audio_base64})

            # Return a response indicating that processing is in progress
            return jsonify({"message": "Processing in progress. Check Socket.IO for results."}), 200
        
        else:
            return jsonify({"error": "Invalid content type. Please use 'application/json'."}), 415
    except KeyError as ke:
        return jsonify({"error": f"KeyError: {str(ke)}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
     socketio.run(app, host='0.0.0.0', port=3200)