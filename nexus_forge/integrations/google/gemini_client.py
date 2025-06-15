"""
Gemini 2.5 Pro/Flash Integration Module - Advanced Text Generation
"""

import asyncio
import os
from typing import Dict, Any, List, Optional, AsyncIterator
import aiohttp
import google.generativeai as genai
import logging
import json
import time
from google.api_core import exceptions as google_exceptions

from nexus_forge.core.error_handling import (
    handle_ai_service_error_async,
    CircuitBreaker,
    retry,
    async_retry
)
from nexus_forge.core.exceptions import (
    ServiceUnavailableError,
    RateLimitError,
    AuthenticationError,
    ModelNotFoundError,
    ModelTimeoutError,
    InvalidAPIKeyError,
    QuotaExceededError,
    PayloadTooLargeError
)
from nexus_forge.core.monitoring import create_ai_operation_monitor
from nexus_forge.core.caching_decorators import (
    cache_ai_response, 
    CacheStrategy, 
    semantic_cache,
    conditional_cache
)

logger = logging.getLogger(__name__)

# Create monitoring decorator for this service
monitor_operation = create_ai_operation_monitor("gemini")

class GeminiClient:
    """Production Gemini 2.5 Pro/Flash client with comprehensive error handling"""
    
    def __init__(self, project_id: str, location: str = "us-central1", api_key: Optional[str] = None):
        self.project_id = project_id
        self.location = location
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "AIzaSyA5YfZt0dp3ncJ2V-G6iEEhwq9IfjlxkDY")
        self.service_name = "gemini"
        
        # Initialize circuit breaker for this client
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=60,
            expected_exception=(ServiceUnavailableError, RateLimitError)
        )
        
        # Rate limiting and quota tracking
        self._request_count = 0
        self._last_request_time = 0
        self._min_request_interval = 0.1  # 100ms between requests
        
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Gemini API client with error handling"""
        try:
            if not self.api_key:
                raise InvalidAPIKeyError(
                    "Gemini API key not provided",
                    service="gemini"
                )
            
            # Configure Gemini API
            genai.configure(api_key=self.api_key)
            
            # Test API key validity
            self._test_api_connection()
            
            # Initialize models
            self.pro_model = genai.GenerativeModel('gemini-2.0-flash-exp')
            self.flash_model = genai.GenerativeModel('gemini-2.0-flash-exp')
            
            # Configure generation settings
            self.generation_config = {
                "max_output_tokens": 8192,
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 40
            }
            
            logger.info("Gemini client initialized successfully with API key")
            
        except google_exceptions.Unauthenticated as e:
            raise InvalidAPIKeyError(
                f"Invalid Gemini API key: {str(e)}",
                service="gemini"
            ) from e
        except google_exceptions.PermissionDenied as e:
            raise AuthenticationError(
                f"Permission denied for Gemini API: {str(e)}",
                service="gemini"
            ) from e
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {str(e)}")
            raise ServiceUnavailableError(
                f"Failed to initialize Gemini client: {str(e)}",
                service="gemini"
            ) from e
    
    def _test_api_connection(self):
        """Test API connection with a minimal request"""
        try:
            # Simple test to verify API key
            test_model = genai.GenerativeModel('gemini-2.0-flash-exp')
            # This will raise an exception if the API key is invalid
            _ = test_model.count_tokens("test")
        except Exception as e:
            logger.error(f"API connection test failed: {str(e)}")
            raise
    
    def _handle_rate_limiting(self):
        """Handle rate limiting between requests"""
        current_time = time.time()
        time_since_last_request = current_time - self._last_request_time
        
        if time_since_last_request < self._min_request_interval:
            sleep_time = self._min_request_interval - time_since_last_request
            time.sleep(sleep_time)
        
        self._last_request_time = time.time()
        self._request_count += 1
    
    def _handle_gemini_exception(self, e: Exception, operation: str) -> Exception:
        """Convert Gemini API exceptions to our custom exceptions"""
        if isinstance(e, google_exceptions.ResourceExhausted):
            if "quota" in str(e).lower():
                return QuotaExceededError(
                    f"Gemini API quota exceeded during {operation}: {str(e)}",
                    quota_type="requests"
                )
            else:
                return RateLimitError(
                    f"Gemini API rate limit exceeded during {operation}: {str(e)}",
                    retry_after=60  # Standard retry after 1 minute
                )
        elif isinstance(e, google_exceptions.Unauthenticated):
            return InvalidAPIKeyError(
                f"Invalid API key during {operation}: {str(e)}",
                service="gemini"
            )
        elif isinstance(e, google_exceptions.PermissionDenied):
            return AuthenticationError(
                f"Permission denied during {operation}: {str(e)}",
                service="gemini"
            )
        elif isinstance(e, google_exceptions.NotFound):
            return ModelNotFoundError(
                f"Gemini model not found during {operation}: {str(e)}",
                model="gemini-2.0-flash-exp"
            )
        elif isinstance(e, google_exceptions.DeadlineExceeded):
            return ModelTimeoutError(
                f"Gemini request timeout during {operation}: {str(e)}",
                model="gemini-2.0-flash-exp",
                timeout=30.0
            )
        elif isinstance(e, google_exceptions.InvalidArgument):
            if "too large" in str(e).lower():
                return PayloadTooLargeError(
                    f"Request payload too large during {operation}: {str(e)}"
                )
            else:
                return ServiceUnavailableError(
                    f"Invalid request during {operation}: {str(e)}",
                    service="gemini"
                )
        else:
            return ServiceUnavailableError(
                f"Gemini service error during {operation}: {str(e)}",
                service="gemini"
            )
    
    @handle_ai_service_error_async
    @create_ai_operation_monitor("gemini")
    @cache_ai_response(
        ttl=None,  # Uses AI-specific TTL based on content type
        strategy=CacheStrategy.COMPRESSED,  # Compress large responses
        cache_tag="gemini_content",
        ignore_args=["stream"]  # Don't include stream param in cache key
    )
    async def generate_content(
        self,
        prompt: str,
        model_type: str = "flash",
        generation_config: Optional[Dict[str, Any]] = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Generate content using Gemini models with comprehensive error handling
        
        Args:
            prompt: Input prompt
            model_type: "pro" or "flash"
            generation_config: Override generation settings
            stream: Whether to stream response
            
        Returns:
            Generated content and metadata
            
        Raises:
            ServiceUnavailableError: When service is down
            RateLimitError: When rate limits are exceeded
            AuthenticationError: When API key is invalid
            ModelTimeoutError: When request times out
            PayloadTooLargeError: When prompt is too large
        """
        
        # Validate inputs
        if not prompt or not prompt.strip():
            raise PayloadTooLargeError("Prompt cannot be empty")
        
        if len(prompt) > 1000000:  # 1MB limit
            raise PayloadTooLargeError(
                f"Prompt too large: {len(prompt)} characters",
                size=len(prompt),
                max_size=1000000
            )
        
        model = self.pro_model if model_type == "pro" else self.flash_model
        config = generation_config or self.generation_config
        
        # Apply rate limiting
        self._handle_rate_limiting()
        
        try:
            # Execute with circuit breaker protection
            if stream:
                return await self.circuit_breaker.async_call(
                    self._generate_streaming, model, prompt, config
                )
            else:
                return await self.circuit_breaker.async_call(
                    self._generate_standard, model, prompt, config
                )
                
        except Exception as e:
            # Convert to our custom exceptions
            custom_exception = self._handle_gemini_exception(e, "generate_content")
            logger.error(f"Gemini generation failed: {str(custom_exception)}")
            raise custom_exception from e
    
    @async_retry(
        max_attempts=3,
        exponential=True,
        exceptions=(ServiceUnavailableError, RateLimitError)
    )
    async def _generate_standard(
        self,
        model: genai.GenerativeModel,
        prompt: str,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate content in standard mode with retries"""
        
        try:
            # Add timeout for the generation
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    model.generate_content,
                    prompt,
                    generation_config=genai.types.GenerationConfig(**config)
                ),
                timeout=30.0  # 30 second timeout
            )
            
            # Check if response was blocked
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'finish_reason'):
                    finish_reason = str(candidate.finish_reason)
                    if finish_reason in ['SAFETY', 'RECITATION', 'OTHER']:
                        raise ServiceUnavailableError(
                            f"Content generation blocked: {finish_reason}",
                            service="gemini"
                        )
            
            # Extract safety ratings safely
            safety_ratings = []
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'safety_ratings'):
                    safety_ratings = [
                        {
                            "category": str(rating.category),
                            "probability": str(rating.probability),
                            "blocked": getattr(rating, 'blocked', False)
                        }
                        for rating in candidate.safety_ratings
                    ]
            
            # Extract usage metadata safely
            usage_metadata = {}
            if hasattr(response, 'usage_metadata'):
                usage_metadata = {
                    "prompt_token_count": getattr(response.usage_metadata, 'prompt_token_count', 0),
                    "candidates_token_count": getattr(response.usage_metadata, 'candidates_token_count', 0),
                    "total_token_count": getattr(response.usage_metadata, 'total_token_count', 0)
                }
            
            # Validate response content
            content = getattr(response, 'text', '')
            if not content:
                raise ServiceUnavailableError(
                    "Empty response from Gemini API",
                    service="gemini"
                )
            
            return {
                "content": content,
                "usage_metadata": usage_metadata,
                "safety_ratings": safety_ratings,
                "model_type": "gemini-2.0-flash-exp"
            }
            
        except asyncio.TimeoutError as e:
            raise ModelTimeoutError(
                "Gemini API request timeout",
                model="gemini-2.0-flash-exp",
                timeout=30.0
            ) from e
        except Exception as e:
            # Let the exception handler in the calling method handle this
            raise
    
    @async_retry(
        max_attempts=3,
        exponential=True,
        exceptions=(ServiceUnavailableError, RateLimitError)
    )
    async def _generate_streaming(
        self,
        model: genai.GenerativeModel,
        prompt: str,
        config: Dict[str, Any]
    ) -> AsyncIterator[Dict[str, Any]]:
        """Generate content in streaming mode with error handling"""
        
        async def stream_generator():
            try:
                response_stream = await asyncio.wait_for(
                    asyncio.to_thread(
                        model.generate_content,
                        prompt,
                        generation_config=genai.types.GenerationConfig(**config),
                        stream=True
                    ),
                    timeout=30.0
                )
                
                chunk_count = 0
                for chunk in response_stream:
                    chunk_count += 1
                    if hasattr(chunk, 'text') and chunk.text:
                        yield {
                            "content": chunk.text,
                            "is_final": False,
                            "chunk_index": chunk_count
                        }
                    
                    # Safety check - don't stream indefinitely
                    if chunk_count > 1000:
                        logger.warning("Streaming response exceeded maximum chunks")
                        break
                
                # Final chunk with metadata
                yield {
                    "content": "",
                    "is_final": True,
                    "chunk_count": chunk_count,
                    "usage_metadata": {
                        "total_token_count": 0  # Streaming doesn't provide final token count
                    }
                }
                
            except asyncio.TimeoutError as e:
                raise ModelTimeoutError(
                    "Gemini streaming request timeout",
                    model="gemini-2.0-flash-exp",
                    timeout=30.0
                ) from e
            except Exception as e:
                # Let the calling method handle exception conversion
                raise
        
        return stream_generator()
    
    @handle_ai_service_error_async
    @conditional_cache(
        condition_func=lambda result: result is not None,
        ttl=1800,  # 30 minutes for chat sessions
        strategy=CacheStrategy.SIMPLE
    )
    async def create_chat_session(
        self,
        system_instruction: Optional[str] = None,
        model_type: str = "flash"
    ) -> genai.ChatSession:
        """Create a chat session for multi-turn conversations with error handling"""
        
        try:
            model = self.pro_model if model_type == "pro" else self.flash_model
            
            if system_instruction:
                if len(system_instruction) > 50000:  # Reasonable limit for system instruction
                    raise PayloadTooLargeError(
                        f"System instruction too large: {len(system_instruction)} characters",
                        size=len(system_instruction),
                        max_size=50000
                    )
                
                model = genai.GenerativeModel(
                    model_name='gemini-2.0-flash-exp',
                    system_instruction=system_instruction
                )
            
            chat_session = model.start_chat()
            logger.info(f"Created Gemini chat session with {model_type} model")
            return chat_session
            
        except Exception as e:
            custom_exception = self._handle_gemini_exception(e, "create_chat_session")
            logger.error(f"Failed to create chat session: {str(custom_exception)}")
            raise custom_exception from e
    
    @handle_ai_service_error_async
    @create_ai_operation_monitor("gemini")
    @semantic_cache(
        similarity_threshold=0.85,  # High similarity for code generation
        ttl=86400  # 24 hours for code
    )
    async def generate_code(
        self,
        requirements: str,
        language: str = "python",
        framework: Optional[str] = None
    ) -> Dict[str, Any]:
        """Specialized method for code generation with error handling"""
        
        # Validate inputs
        if not requirements or not requirements.strip():
            raise PayloadTooLargeError("Code requirements cannot be empty")
        
        framework_text = f" using {framework}" if framework else ""
        
        prompt = f"""
        Generate high-quality {language} code{framework_text} based on these requirements:
        
        {requirements}
        
        Requirements:
        - Follow best practices and conventions
        - Include proper error handling
        - Add comprehensive docstrings/comments
        - Use type hints (if applicable)
        - Ensure code is production-ready
        - Include example usage if appropriate
        """
        
        try:
            result = await self.generate_content(
                prompt=prompt,
                model_type="pro",  # Use Pro for code generation
                generation_config={
                    "max_output_tokens": 8192,
                    "temperature": 0.3,  # Lower temperature for code
                    "top_p": 0.8
                }
            )
            
            # Add code-specific metadata
            result["generation_type"] = "code"
            result["language"] = language
            result["framework"] = framework
            
            return result
            
        except Exception as e:
            logger.error(f"Code generation failed: {str(e)}")
            raise
    
    @handle_ai_service_error_async
    @cache_ai_response(
        ttl=43200,  # 12 hours for app specs
        strategy=CacheStrategy.COMPRESSED,
        cache_tag="app_specifications"
    )
    async def generate_app_specification(
        self,
        app_description: str,
        requirements: List[str],
        target_platform: str = "web"
    ) -> Dict[str, Any]:
        """Generate comprehensive app specification with error handling"""
        
        # Validate inputs
        if not app_description or not app_description.strip():
            raise PayloadTooLargeError("App description cannot be empty")
        
        if not requirements:
            raise PayloadTooLargeError("Requirements list cannot be empty")
        
        try:
            requirements_text = "\n".join([f"- {req}" for req in requirements])
            
            prompt = f"""
            Create a comprehensive technical specification for the following application:
            
            App Description: {app_description}
            Target Platform: {target_platform}
            
            Requirements:
            {requirements_text}
            
            Please provide a detailed specification including:
            1. Technical Architecture
            2. Database Schema
            3. API Endpoints
            4. UI/UX Components
            5. Security Considerations
            6. Deployment Strategy
            7. Testing Strategy
            8. Performance Requirements
            
            Format the response as structured JSON with clear sections.
            """
            
            result = await self.generate_content(
                prompt=prompt,
                model_type="pro",
                generation_config={
                    "max_output_tokens": 8192,
                    "temperature": 0.4
                }
            )
            
            # Add specification metadata
            result["generation_type"] = "app_specification"
            result["target_platform"] = target_platform
            result["requirements_count"] = len(requirements)
            
            return result
            
        except Exception as e:
            logger.error(f"App specification generation failed: {str(e)}")
            raise
    
    @handle_ai_service_error_async
    @cache_ai_response(
        ttl=7200,  # 2 hours for prompt optimization
        strategy=CacheStrategy.SIMPLE,
        cache_tag="prompt_optimization"
    )
    async def optimize_prompt_for_service(
        self,
        service_name: str,
        user_prompt: str,
        context: Dict[str, Any]
    ) -> str:
        """Optimize prompts for specific AI services with error handling"""
        
        # Validate inputs
        if not service_name or not user_prompt:
            raise PayloadTooLargeError("Service name and user prompt are required")
        
        if len(user_prompt) > 100000:  # 100KB limit
            raise PayloadTooLargeError(
                f"User prompt too large: {len(user_prompt)} characters",
                size=len(user_prompt),
                max_size=100000
            )
        
        try:
            optimization_prompt = f"""
            Optimize the following user prompt for {service_name} AI service:
            
            Original prompt: {user_prompt}
            
            Context: {json.dumps(context, indent=2)}
            
            Service-specific requirements for {service_name}:
            """
            
            if service_name.lower() == "imagen":
                optimization_prompt += """
                - Be specific about visual elements, style, colors
                - Include composition and layout details
                - Mention art style, lighting, mood
                - Specify resolution and aspect ratio preferences
                - Include quality and detail requirements
                """
            elif service_name.lower() == "veo":
                optimization_prompt += """
                - Describe scene progression and timing
                - Include camera movements and angles
                - Specify visual style and mood
                - Mention transitions and effects
                - Include duration and pacing details
                """
            elif service_name.lower() == "jules":
                optimization_prompt += """
                - Be specific about the coding task
                - Include file names and locations
                - Specify testing requirements
                - Mention coding standards and patterns
                - Include acceptance criteria
                """
            
            optimization_prompt += """
            
            Return an optimized prompt that will produce better results from the target service.
            Keep the user's intent but enhance with service-specific details.
            """
            
            result = await self.generate_content(
                prompt=optimization_prompt,
                model_type="pro",
                generation_config={
                    "temperature": 0.3,
                    "max_output_tokens": 2048
                }
            )
            
            return result["content"]
            
        except Exception as e:
            logger.error(f"Prompt optimization failed for {service_name}: {str(e)}")
            raise
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on Gemini service"""
        try:
            start_time = time.time()
            
            # Simple test generation
            result = await self.generate_content(
                prompt="Test prompt for health check",
                model_type="flash",
                generation_config={
                    "max_output_tokens": 10,
                    "temperature": 0.1
                }
            )
            
            latency = time.time() - start_time
            
            return {
                "status": "healthy",
                "latency_ms": round(latency * 1000, 2),
                "circuit_breaker_state": self.circuit_breaker.state.value,
                "request_count": self._request_count,
                "timestamp": time.time()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "circuit_breaker_state": self.circuit_breaker.state.value,
                "timestamp": time.time()
            }

# Example usage functions with error handling
async def example_gemini_usage():
    """Example usage of the Gemini client with proper error handling"""
    
    try:
        client = GeminiClient(project_id="your-project-id")
        
        # Health check
        health = await client.health_check()
        print(f"Health check: {health}")
        
        # Standard generation
        result = await client.generate_content(
            prompt="Explain the benefits of microservices architecture",
            model_type="flash"
        )
        print(result["content"])
        
        # Code generation
        code_result = await client.generate_code(
            requirements="Create a FastAPI endpoint for user authentication with JWT tokens",
            language="python",
            framework="FastAPI"
        )
        print(code_result["content"])
        
        # Chat session
        chat = await client.create_chat_session(
            system_instruction="You are a helpful coding assistant specializing in Python and web development."
        )
        
        response = await asyncio.to_thread(
            chat.send_message,
            "How can I optimize this database query?"
        )
        print(response.text)
        
        # App specification generation
        app_spec = await client.generate_app_specification(
            app_description="A task management application for teams",
            requirements=[
                "Real-time collaboration",
                "User authentication",
                "File attachments",
                "Mobile responsive"
            ],
            target_platform="web"
        )
        print(app_spec["content"])
        
    except InvalidAPIKeyError as e:
        print(f"API key error: {e.message}")
    except RateLimitError as e:
        print(f"Rate limit exceeded: {e.message}. Retry after: {e.details.get('retry_after')} seconds")
    except ServiceUnavailableError as e:
        print(f"Service unavailable: {e.message}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(example_gemini_usage())