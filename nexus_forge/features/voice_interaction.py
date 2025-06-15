"""
Voice interaction feature for Parallax Pal

Enables voice-based research queries using Google Cloud Speech-to-Text
and Text-to-Speech APIs with ADK integration.
"""

from google.cloud import speech_v1, texttospeech_v1
from google.adk.streaming import AudioStreamingSession
import base64
import asyncio
import logging
from typing import Optional, Dict, Any, AsyncGenerator
from datetime import datetime
import io
import wave
import json

logger = logging.getLogger(__name__)


class VoiceInteractionHandler:
    """Enable voice-based research queries with Starri"""
    
    def __init__(self, adk_integration):
        """
        Initialize voice interaction handler
        
        Args:
            adk_integration: ParallaxPalADK instance
        """
        self.adk = adk_integration
        
        # Initialize Google Cloud clients
        self.speech_client = speech_v1.SpeechAsyncClient()
        self.tts_client = texttospeech_v1.TextToSpeechAsyncClient()
        
        # Voice configurations
        self.voice_configs = {
            'starri_default': {
                'name': 'en-US-Neural2-F',
                'language_code': 'en-US',
                'ssml_gender': texttospeech_v1.SsmlVoiceGender.FEMALE,
                'speaking_rate': 1.1,
                'pitch': 1.0
            },
            'starri_excited': {
                'name': 'en-US-Neural2-F',
                'language_code': 'en-US',
                'ssml_gender': texttospeech_v1.SsmlVoiceGender.FEMALE,
                'speaking_rate': 1.3,
                'pitch': 2.0
            },
            'starri_thoughtful': {
                'name': 'en-US-Neural2-F',
                'language_code': 'en-US',
                'ssml_gender': texttospeech_v1.SsmlVoiceGender.FEMALE,
                'speaking_rate': 0.9,
                'pitch': -1.0
            }
        }
        
        # Audio streaming sessions
        self.active_sessions: Dict[str, AudioStreamingSession] = {}
        
        logger.info("Voice interaction handler initialized")
    
    async def process_voice_query(
        self, 
        audio_data: bytes,
        audio_format: str = "webm",
        language: str = "en-US",
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Convert voice to text and process query
        
        Args:
            audio_data: Raw audio data
            audio_format: Audio format (webm, wav, mp3)
            language: Language code
            session_id: Optional session ID for context
            
        Returns:
            Dict with transcript and confidence
        """
        try:
            # Configure speech recognition based on format
            encoding_map = {
                'webm': speech_v1.RecognitionConfig.AudioEncoding.WEBM_OPUS,
                'wav': speech_v1.RecognitionConfig.AudioEncoding.LINEAR16,
                'mp3': speech_v1.RecognitionConfig.AudioEncoding.MP3,
                'ogg': speech_v1.RecognitionConfig.AudioEncoding.OGG_OPUS
            }
            
            config = speech_v1.RecognitionConfig(
                encoding=encoding_map.get(audio_format, speech_v1.RecognitionConfig.AudioEncoding.WEBM_OPUS),
                sample_rate_hertz=48000 if audio_format == 'webm' else 16000,
                language_code=language,
                enable_automatic_punctuation=True,
                enable_word_time_offsets=True,
                model="latest_long",
                use_enhanced=True,
                # Enable speaker diarization for multi-speaker scenarios
                enable_speaker_diarization=True,
                diarization_speaker_count=2,
                # Alternative languages for better recognition
                alternative_language_codes=['en-GB', 'en-AU']
            )
            
            # Create recognition request
            audio = speech_v1.RecognitionAudio(content=audio_data)
            
            # Perform recognition
            response = await self.speech_client.recognize(
                config=config,
                audio=audio
            )
            
            # Process results
            if not response.results:
                return {
                    'success': False,
                    'error': 'No speech detected',
                    'transcript': ''
                }
            
            # Get best transcript
            best_alternative = response.results[0].alternatives[0]
            
            # Extract word timings for animation sync
            word_timings = []
            for word in best_alternative.words:
                word_timings.append({
                    'word': word.word,
                    'start_time': word.start_time.total_seconds(),
                    'end_time': word.end_time.total_seconds()
                })
            
            result = {
                'success': True,
                'transcript': best_alternative.transcript,
                'confidence': best_alternative.confidence,
                'language': response.results[0].language_code,
                'word_timings': word_timings,
                'is_final': True
            }
            
            # Log successful transcription
            logger.info(
                f"Voice transcription successful: {len(best_alternative.transcript)} chars, "
                f"confidence: {best_alternative.confidence:.2f}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Speech recognition error: {e}")
            return {
                'success': False,
                'error': str(e),
                'transcript': ''
            }
    
    async def generate_audio_response(
        self,
        text: str,
        emotion: str = "default",
        format: str = "mp3",
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Convert text response to speech with Starri's voice
        
        Args:
            text: Text to convert
            emotion: Emotion state (default, excited, thoughtful)
            format: Audio format (mp3, wav, ogg)
            session_id: Optional session ID
            
        Returns:
            Dict with audio data and metadata
        """
        try:
            # Get voice config based on emotion
            voice_config = self.voice_configs.get(
                f'starri_{emotion}',
                self.voice_configs['starri_default']
            )
            
            # Prepare SSML for more natural speech
            ssml_text = self._create_ssml(text, emotion)
            
            synthesis_input = texttospeech_v1.SynthesisInput(ssml=ssml_text)
            
            voice = texttospeech_v1.VoiceSelectionParams(
                language_code=voice_config['language_code'],
                name=voice_config['name'],
                ssml_gender=voice_config['ssml_gender']
            )
            
            # Audio format configuration
            audio_encoding_map = {
                'mp3': texttospeech_v1.AudioEncoding.MP3,
                'wav': texttospeech_v1.AudioEncoding.LINEAR16,
                'ogg': texttospeech_v1.AudioEncoding.OGG_OPUS
            }
            
            audio_config = texttospeech_v1.AudioConfig(
                audio_encoding=audio_encoding_map.get(format, texttospeech_v1.AudioEncoding.MP3),
                speaking_rate=voice_config['speaking_rate'],
                pitch=voice_config['pitch'],
                volume_gain_db=0.0,
                sample_rate_hertz=24000,
                effects_profile_id=['headphone-class-device']  # Optimize for headphones
            )
            
            # Generate speech
            response = await self.tts_client.synthesize_speech(
                input=synthesis_input,
                voice=voice,
                audio_config=audio_config
            )
            
            # Calculate audio duration
            audio_duration = self._calculate_audio_duration(
                response.audio_content,
                format
            )
            
            result = {
                'success': True,
                'audio_data': base64.b64encode(response.audio_content).decode('utf-8'),
                'format': format,
                'duration_seconds': audio_duration,
                'text_length': len(text),
                'emotion': emotion,
                'voice': voice_config['name']
            }
            
            logger.info(
                f"TTS generation successful: {len(text)} chars, "
                f"duration: {audio_duration:.2f}s"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"TTS generation error: {e}")
            return {
                'success': False,
                'error': str(e),
                'audio_data': ''
            }
    
    async def create_streaming_session(
        self,
        session_id: str,
        user_id: str,
        language: str = "en-US"
    ) -> AudioStreamingSession:
        """
        Create bidirectional audio streaming session
        
        Args:
            session_id: WebSocket session ID
            user_id: User ID
            language: Language code
            
        Returns:
            AudioStreamingSession instance
        """
        # Create ADK audio streaming session
        session = AudioStreamingSession(
            app=self.adk.app,
            user_id=user_id,
            session_id=session_id,
            config={
                'language': language,
                'enable_interim_results': True,
                'single_utterance': False,
                'voice_config': self.voice_configs['starri_default']
            }
        )
        
        # Store session
        self.active_sessions[session_id] = session
        
        logger.info(f"Audio streaming session created: {session_id}")
        
        return session
    
    async def process_audio_stream(
        self,
        session_id: str,
        audio_chunk: bytes
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Process audio stream chunk
        
        Args:
            session_id: Session ID
            audio_chunk: Audio data chunk
            
        Yields:
            Stream processing results
        """
        session = self.active_sessions.get(session_id)
        if not session:
            yield {
                'type': 'error',
                'error': 'No active streaming session'
            }
            return
        
        # Process audio chunk through streaming session
        async for event in session.process_audio(audio_chunk):
            if event.type == 'transcript':
                yield {
                    'type': 'interim_transcript' if not event.is_final else 'final_transcript',
                    'transcript': event.transcript,
                    'confidence': event.confidence,
                    'is_final': event.is_final
                }
            elif event.type == 'response_audio':
                yield {
                    'type': 'audio_response',
                    'audio_data': base64.b64encode(event.audio).decode('utf-8'),
                    'text': event.text,
                    'emotion': event.metadata.get('emotion', 'default')
                }
            elif event.type == 'research_update':
                yield {
                    'type': 'research_progress',
                    'agent': event.agent,
                    'progress': event.progress,
                    'status': event.status
                }
    
    async def close_streaming_session(self, session_id: str):
        """Close and cleanup streaming session"""
        
        session = self.active_sessions.pop(session_id, None)
        if session:
            await session.close()
            logger.info(f"Audio streaming session closed: {session_id}")
    
    def _create_ssml(self, text: str, emotion: str) -> str:
        """
        Create SSML markup for more natural speech
        
        Args:
            text: Plain text
            emotion: Emotion state
            
        Returns:
            SSML formatted text
        """
        # Add pauses for natural speech
        text = text.replace('. ', '. <break time="300ms"/>')
        text = text.replace(', ', ', <break time="200ms"/>')
        text = text.replace('? ', '? <break time="400ms"/>')
        text = text.replace('! ', '! <break time="400ms"/>')
        
        # Add emphasis based on emotion
        if emotion == 'excited':
            # Add emphasis to key words
            emphasis_words = ['amazing', 'fantastic', 'incredible', 'discovered', 'found']
            for word in emphasis_words:
                text = text.replace(
                    f' {word} ',
                    f' <emphasis level="strong">{word}</emphasis> '
                )
        elif emotion == 'thoughtful':
            # Add longer pauses
            text = text.replace('. ', '. <break time="500ms"/>')
        
        # Wrap in SSML
        ssml = f"""<speak>
            <prosody rate="{100 if emotion != 'excited' else 110}%">
                {text}
            </prosody>
        </speak>"""
        
        return ssml
    
    def _calculate_audio_duration(self, audio_data: bytes, format: str) -> float:
        """
        Calculate audio duration from audio data
        
        Args:
            audio_data: Raw audio bytes
            format: Audio format
            
        Returns:
            Duration in seconds
        """
        try:
            if format == 'wav':
                # Parse WAV header
                with io.BytesIO(audio_data) as audio_io:
                    with wave.open(audio_io, 'rb') as wav_file:
                        frames = wav_file.getnframes()
                        rate = wav_file.getframerate()
                        return frames / float(rate)
            else:
                # Estimate based on bitrate
                # MP3: ~128kbps, OGG: ~96kbps
                bitrate = 128000 if format == 'mp3' else 96000
                return len(audio_data) * 8 / bitrate
        except Exception:
            # Fallback estimation
            return len(audio_data) / 20000  # Rough estimate
    
    async def generate_voice_feedback(
        self,
        feedback_type: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate contextual voice feedback
        
        Args:
            feedback_type: Type of feedback
            context: Optional context data
            
        Returns:
            Audio response
        """
        feedback_templates = {
            'greeting': [
                "Hi there! I'm Starri, your research assistant. What would you like to explore today?",
                "Hello! Ready to discover something amazing together?",
                "Welcome back! I'm excited to help you research today!"
            ],
            'thinking': [
                "Hmm, let me think about that for a moment...",
                "That's an interesting question. Let me search for information...",
                "Great question! I'm coordinating my agents to find the best answers..."
            ],
            'found_results': [
                "I found some fascinating information for you!",
                "Great news! I discovered several relevant sources.",
                "My agents have compiled some excellent findings!"
            ],
            'error': [
                "Oops, I encountered a small hiccup. Let me try again...",
                "I'm having trouble with that request. Could you rephrase it?",
                "Something went wrong, but don't worry, I'm here to help!"
            ],
            'complete': [
                "All done! I've compiled a comprehensive report for you.",
                "Research complete! Take a look at what I found.",
                "Finished! I hope this information is helpful."
            ]
        }
        
        # Select appropriate template
        templates = feedback_templates.get(feedback_type, feedback_templates['thinking'])
        import random
        text = random.choice(templates)
        
        # Determine emotion based on feedback type
        emotion_map = {
            'greeting': 'excited',
            'thinking': 'thoughtful',
            'found_results': 'excited',
            'error': 'thoughtful',
            'complete': 'default'
        }
        
        emotion = emotion_map.get(feedback_type, 'default')
        
        # Generate audio
        return await self.generate_audio_response(text, emotion)