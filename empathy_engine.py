
import os
import re
import json
import tempfile
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from flask import Flask, render_template_string, request, jsonify, send_file
import pyttsx3
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import threading
import time

@dataclass
class VoiceParameters:
    """Data class to hold voice modulation parameters"""
    rate: int = 200        # Words per minute
    pitch: float = 0.0     # Pitch adjustment (-50 to 50)
    volume: float = 0.9    # Volume level (0.0 to 1.0)
    emphasis: str = "none" # Emphasis level for SSML

@dataclass
class EmotionResult:
    """Data class to hold emotion detection results"""
    emotion: str
    confidence: float
    intensity: float  # 0.0 to 1.0

class EmotionDetector:
    """Advanced emotion detection using Hugging Face transformers"""
    
    def __init__(self):
        # Load pre-trained emotion classification model
        try:
            self.classifier = pipeline(
                "text-classification", 
                model="j-hartmann/emotion-english-distilroberta-base",
                device=-1  # Use CPU
            )
            self.emotions_map = {
                'joy': 'happy',
                'sadness': 'sad', 
                'anger': 'angry',
                'fear': 'worried',
                'surprise': 'surprised',
                'disgust': 'disgusted',
                'neutral': 'neutral'
            }
        except Exception as e:
            print(f"Warning: Could not load advanced model, falling back to basic sentiment: {e}")
            # Fallback to basic sentiment
            self.classifier = None
            
    def detect_emotion(self, text: str) -> EmotionResult:
        """Detect emotion in text with confidence and intensity"""
        if self.classifier:
            try:
                result = self.classifier(text)
                emotion_raw = result[0]['label'].lower()
                confidence = result[0]['score']
                
                # Map to our emotion categories
                emotion = self.emotions_map.get(emotion_raw, 'neutral')
                
                # Calculate intensity based on confidence and text markers
                intensity = self._calculate_intensity(text, confidence)
                
                return EmotionResult(emotion=emotion, confidence=confidence, intensity=intensity)
            except Exception as e:
                print(f"Error in emotion detection: {e}")
                
        # Fallback to basic sentiment analysis
        return self._basic_sentiment_analysis(text)
    
    def _calculate_intensity(self, text: str, base_confidence: float) -> float:
        """Calculate emotion intensity based on text markers and confidence"""
        intensity_markers = {
            'high': ['!!!', 'amazing', 'incredible', 'fantastic', 'terrible', 'awful', 'furious', 'ecstatic'],
            'medium': ['!', 'great', 'good', 'bad', 'upset', 'happy', 'sad'],
            'low': ['.', 'okay', 'fine', 'alright']
        }
        
        text_lower = text.lower()
        
        # Check for intensity markers
        intensity_score = base_confidence
        
        for marker in intensity_markers['high']:
            if marker in text_lower:
                intensity_score = min(1.0, intensity_score + 0.3)
                
        for marker in intensity_markers['medium']:
            if marker in text_lower:
                intensity_score = min(1.0, intensity_score + 0.1)
        
        # Check for caps (shouting)
        caps_ratio = sum(1 for c in text if c.isupper()) / max(1, len(text))
        if caps_ratio > 0.3:
            intensity_score = min(1.0, intensity_score + 0.2)
            
        return max(0.1, min(1.0, intensity_score))
    
    def _basic_sentiment_analysis(self, text: str) -> EmotionResult:
        """Fallback basic sentiment analysis"""
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love', 'happy']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'sad', 'angry', 'frustrated', 'annoying']
        
        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        if pos_count > neg_count:
            return EmotionResult(emotion='happy', confidence=0.7, intensity=0.6)
        elif neg_count > pos_count:
            return EmotionResult(emotion='angry', confidence=0.7, intensity=0.6)
        else:
            return EmotionResult(emotion='neutral', confidence=0.8, intensity=0.3)

class EmpathyEngine:
    """Main engine for emotion-aware text-to-speech"""
    
    def __init__(self):
        self.emotion_detector = EmotionDetector()
        self.voice_mappings = self._initialize_voice_mappings()
        self._init_tts_engine()
    
    def _init_tts_engine(self):
        """Initialize TTS engine with thread safety"""
        self.tts_lock = threading.Lock()
        
    def _initialize_voice_mappings(self) -> Dict[str, VoiceParameters]:
        """Initialize emotion to voice parameter mappings"""
        return {
            'happy': VoiceParameters(rate=220, pitch=10, volume=0.95, emphasis="strong"),
            'excited': VoiceParameters(rate=240, pitch=15, volume=1.0, emphasis="strong"),
            'sad': VoiceParameters(rate=160, pitch=-10, volume=0.7, emphasis="none"),
            'angry': VoiceParameters(rate=200, pitch=5, volume=0.9, emphasis="strong"),
            'worried': VoiceParameters(rate=180, pitch=-5, volume=0.8, emphasis="moderate"),
            'surprised': VoiceParameters(rate=210, pitch=12, volume=0.9, emphasis="strong"),
            'disgusted': VoiceParameters(rate=170, pitch=-8, volume=0.8, emphasis="none"),
            'neutral': VoiceParameters(rate=200, pitch=0, volume=0.85, emphasis="none"),
        }
    
    def process_text(self, text: str) -> Tuple[EmotionResult, VoiceParameters, str]:
        """Process text and return emotion, voice params, and audio file path"""
        
        # Step 1: Detect emotion
        emotion_result = self.emotion_detector.detect_emotion(text)
        
        # Step 2: Map emotion to voice parameters
        base_params = self.voice_mappings.get(emotion_result.emotion, 
                                              self.voice_mappings['neutral'])
        
        # Step 3: Apply intensity scaling
        adjusted_params = self._apply_intensity_scaling(base_params, emotion_result.intensity)
        
        # Step 4: Generate audio
        audio_path = self._generate_audio(text, adjusted_params)
        
        return emotion_result, adjusted_params, audio_path
    
    def _apply_intensity_scaling(self, base_params: VoiceParameters, intensity: float) -> VoiceParameters:
        """Scale voice parameters based on emotion intensity"""
        # Create a copy to avoid modifying the original
        scaled_params = VoiceParameters(
            rate=int(base_params.rate + (base_params.rate - 200) * intensity * 0.5),
            pitch=base_params.pitch * (1 + intensity * 0.5),
            volume=min(1.0, base_params.volume + (intensity - 0.5) * 0.2),
            emphasis=base_params.emphasis if intensity > 0.6 else "none"
        )
        
        # Ensure parameters stay within reasonable bounds
        scaled_params.rate = max(100, min(300, scaled_params.rate))
        scaled_params.pitch = max(-50, min(50, scaled_params.pitch))
        scaled_params.volume = max(0.1, min(1.0, scaled_params.volume))
        
        return scaled_params
    
    def _generate_audio(self, text: str, params: VoiceParameters) -> str:
        """Generate audio file with specified voice parameters"""
        
        with self.tts_lock:  # Thread safety for TTS engine
            try:
                # Create TTS engine
                engine = pyttsx3.init()
                
                # Set voice parameters
                engine.setProperty('rate', params.rate)
                engine.setProperty('volume', params.volume)
                
                # Try to set pitch if supported (platform dependent)
                try:
                    voices = engine.getProperty('voices')
                    if voices:
                        engine.setProperty('voice', voices[0].id)
                except Exception as e:
                    print(f"Note: Pitch adjustment may not be fully supported on this platform: {e}")
                
                # Generate temporary file
                temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                temp_file.close()
                
                # Save to file
                engine.save_to_file(text, temp_file.name)
                engine.runAndWait()
                engine.stop()
                
                return temp_file.name
                
            except Exception as e:
                print(f"Error generating audio: {e}")
                # Return a placeholder path or raise exception
                raise Exception(f"Failed to generate audio: {e}")

# Flask Web Application
app = Flask(__name__)
empathy_engine = EmpathyEngine()

# HTML template for the web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>The Empathy Engine</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: white;
        }
        .container {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
        }
        h1 {
            text-align: center;
            margin-bottom: 10px;
            font-size: 2.5em;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .subtitle {
            text-align: center;
            margin-bottom: 30px;
            opacity: 0.8;
            font-size: 1.1em;
        }
        textarea {
            width: 100%;
            height: 120px;
            border: none;
            border-radius: 15px;
            padding: 15px;
            font-size: 16px;
            background: rgba(255, 255, 255, 0.9);
            color: #333;
            box-sizing: border-box;
            resize: vertical;
        }
        button {
            background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
            color: white;
            border: none;
            padding: 15px 30px;
            border-radius: 25px;
            font-size: 16px;
            cursor: pointer;
            margin-top: 15px;
            transition: transform 0.2s;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0,0,0,0.3);
        }
        button:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        .result {
            margin-top: 30px;
            padding: 20px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            display: none;
        }
        .emotion-info {
            display: flex;
            justify-content: space-between;
            margin-bottom: 15px;
            flex-wrap: wrap;
        }
        .emotion-badge {
            background: rgba(255, 255, 255, 0.2);
            padding: 8px 16px;
            border-radius: 20px;
            margin: 5px;
            font-weight: bold;
        }
        .voice-params {
            background: rgba(0, 0, 0, 0.2);
            padding: 15px;
            border-radius: 10px;
            margin: 15px 0;
        }
        .param-row {
            display: flex;
            justify-content: space-between;
            margin: 8px 0;
        }
        audio {
            width: 100%;
            margin-top: 20px;
            border-radius: 25px;
        }
        .loading {
            text-align: center;
            padding: 20px;
            display: none;
        }
        .spinner {
            border: 4px solid rgba(255, 255, 255, 0.3);
            border-top: 4px solid white;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 15px;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎙️ The Empathy Engine</h1>
        <p class="subtitle">Giving AI a Human Voice - Dynamic emotion-aware text-to-speech</p>
        
        <form id="empathyForm">
            <textarea id="inputText" placeholder="Enter your text here... Try something emotional like 'I'm so excited about this amazing project!' or 'This is really frustrating me.'"></textarea>
            <br>
            <button type="submit" id="processBtn">🎯 Process & Generate Voice</button>
        </form>
        
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p>Analyzing emotions and generating expressive speech...</p>
        </div>
        
        <div class="result" id="result">
            <h3>📊 Analysis Results</h3>
            <div class="emotion-info" id="emotionInfo"></div>
            
            <div class="voice-params" id="voiceParams">
                <h4>🎚️ Voice Parameters Applied</h4>
                <div id="paramsList"></div>
            </div>
            
            <h4>🔊 Generated Audio</h4>
            <audio id="audioPlayer" controls></audio>
        </div>
    </div>

    <script>
        document.getElementById('empathyForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const text = document.getElementById('inputText').value.trim();
            if (!text) {
                alert('Please enter some text to process!');
                return;
            }
            
            // Show loading, hide results
            document.getElementById('loading').style.display = 'block';
            document.getElementById('result').style.display = 'none';
            document.getElementById('processBtn').disabled = true;
            
            try {
                const response = await fetch('/process', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({text: text})
                });
                
                if (!response.ok) {
                    throw new Error('Processing failed');
                }
                
                const data = await response.json();
                
                // Display emotion info
                const emotionInfo = document.getElementById('emotionInfo');
                emotionInfo.innerHTML = `
                    <div class="emotion-badge">Emotion: ${data.emotion.emotion}</div>
                    <div class="emotion-badge">Confidence: ${(data.emotion.confidence * 100).toFixed(1)}%</div>
                    <div class="emotion-badge">Intensity: ${(data.emotion.intensity * 100).toFixed(1)}%</div>
                `;
                
                // Display voice parameters
                const paramsList = document.getElementById('paramsList');
                paramsList.innerHTML = `
                    <div class="param-row"><span>Speech Rate:</span><span>${data.voice_params.rate} WPM</span></div>
                    <div class="param-row"><span>Pitch Adjustment:</span><span>${data.voice_params.pitch > 0 ? '+' : ''}${data.voice_params.pitch}</span></div>
                    <div class="param-row"><span>Volume Level:</span><span>${(data.voice_params.volume * 100).toFixed(0)}%</span></div>
                    <div class="param-row"><span>Emphasis:</span><span>${data.voice_params.emphasis}</span></div>
                `;
                
                // Set up audio player
                const audioPlayer = document.getElementById('audioPlayer');
                audioPlayer.src = `/audio/${data.audio_id}`;
                
                // Show results
                document.getElementById('result').style.display = 'block';
                
            } catch (error) {
                alert('Error processing text: ' + error.message);
            } finally {
                document.getElementById('loading').style.display = 'none';
                document.getElementById('processBtn').disabled = false;
            }
        });
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """Serve the main web interface"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/process', methods=['POST'])
def process_text():
    """API endpoint to process text and generate emotional speech"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Process text through empathy engine
        emotion_result, voice_params, audio_path = empathy_engine.process_text(text)
        
        # Store audio path for retrieval (in production, use proper storage)
        audio_id = str(int(time.time() * 1000))  # Simple ID generation
        app.config[f'audio_{audio_id}'] = audio_path
        
        # Return results
        return jsonify({
            'emotion': {
                'emotion': emotion_result.emotion,
                'confidence': emotion_result.confidence,
                'intensity': emotion_result.intensity
            },
            'voice_params': {
                'rate': voice_params.rate,
                'pitch': voice_params.pitch,
                'volume': voice_params.volume,
                'emphasis': voice_params.emphasis
            },
            'audio_id': audio_id
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/audio/<audio_id>')
def get_audio(audio_id):
    """Serve generated audio files"""
    try:
        audio_path = app.config.get(f'audio_{audio_id}')
        if audio_path and os.path.exists(audio_path):
            return send_file(audio_path, as_attachment=False, mimetype='audio/wav')
        else:
            return "Audio not found", 404
    except Exception as e:
        return f"Error serving audio: {e}", 500

# CLI Interface
def cli_interface():
    """Command line interface for the Empathy Engine"""
    print("🎙️ The Empathy Engine - CLI Mode")
    print("=====================================")
    print("Enter text to convert to emotional speech (or 'quit' to exit)")
    print()
    
    while True:
        try:
            text = input("Enter text: ").strip()
            
            if text.lower() in ['quit', 'exit', 'q']:
                print("Goodbye! 👋")
                break
                
            if not text:
                print("Please enter some text!")
                continue
            
            print("\n🔄 Processing...")
            
            # Process text
            emotion_result, voice_params, audio_path = empathy_engine.process_text(text)
            
            # Display results
            print(f"\n📊 Analysis Results:")
            print(f"   Emotion: {emotion_result.emotion}")
            print(f"   Confidence: {emotion_result.confidence:.2f}")
            print(f"   Intensity: {emotion_result.intensity:.2f}")
            
            print(f"\n🎚️ Voice Parameters:")
            print(f"   Rate: {voice_params.rate} WPM")
            print(f"   Pitch: {voice_params.pitch:+.1f}")
            print(f"   Volume: {voice_params.volume:.1f}")
            print(f"   Emphasis: {voice_params.emphasis}")
            
            print(f"\n🔊 Audio saved to: {audio_path}")
            print("=" * 50)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"Error: {e}")
            print("Please try again.")

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--cli':
        # Run CLI interface
        cli_interface()
    else:
        # Run web interface
        print("🎙️ Starting The Empathy Engine Web Interface...")
        print("📱 Open http://localhost:5000 in your browser")
        print("💡 Or run with '--cli' flag for command line interface")
        print()
        app.run(debug=True, host='0.0.0.0', port=5000)