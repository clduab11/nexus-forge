"""
Pytest configuration and fixtures for Nexus Forge test suite

Provides comprehensive test support for both legacy Parallax Pal features
and new Nexus Forge multi-agent orchestration capabilities.
"""

import asyncio
import os
from typing import Any, AsyncGenerator, Dict, Generator
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
import pytest_asyncio

# Set test environment
os.environ["TESTING"] = "true"
os.environ["ENVIRONMENT"] = "testing"
os.environ["GOOGLE_CLOUD_PROJECT"] = "test-project"
os.environ["GOOGLE_CLOUD_LOCATION"] = "us-central1"
os.environ["GOOGLE_CLOUD_PROJECT_ID"] = "test-project-id"
os.environ["JULES_GITHUB_TOKEN"] = "test-github-token"
os.environ["JULES_WEBHOOK_SECRET"] = "test-webhook-secret"
os.environ["VEO_OUTPUT_BUCKET"] = "test-veo-bucket"
os.environ["DATABASE_URL"] = "sqlite:///test.db"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/tmp/test-credentials.json"

# Override configuration for testing
os.environ["REDIS_PASSWORD"] = "test-password"
os.environ["GOOGLE_CLIENT_ID"] = "test-google-client-id"
os.environ["GOOGLE_CLIENT_SECRET"] = "test-google-client-secret"
os.environ["GOOGLE_REDIRECT_URI"] = "http://localhost:8000/auth/google/callback"
os.environ["GITHUB_CLIENT_ID"] = "test-github-client-id"
os.environ["GITHUB_CLIENT_SECRET"] = "test-github-client-secret"
os.environ["GITHUB_REDIRECT_URI"] = "http://localhost:8000/auth/github/callback"
os.environ["FACEBOOK_CLIENT_ID"] = "test-facebook-client-id"
os.environ["FACEBOOK_CLIENT_SECRET"] = "test-facebook-client-secret"
os.environ["FACEBOOK_REDIRECT_URI"] = "http://localhost:8000/auth/facebook/callback"
os.environ["INSTAGRAM_CLIENT_ID"] = "test-instagram-client-id"
os.environ["INSTAGRAM_CLIENT_SECRET"] = "test-instagram-client-secret"
os.environ["INSTAGRAM_REDIRECT_URI"] = "http://localhost:8000/auth/instagram/callback"
os.environ["SMTP_HOST"] = "smtp.test.com"
os.environ["SMTP_PORT"] = "587"
os.environ["SMTP_USER"] = "test@test.com"
os.environ["SMTP_PASSWORD"] = "test-password"
os.environ["SMTP_FROM_EMAIL"] = "noreply@test.com"
os.environ["STRIPE_SECRET_KEY"] = "sk_test_test_key"
os.environ["STRIPE_PUBLISHABLE_KEY"] = "pk_test_test_key"
os.environ["STRIPE_WEBHOOK_SECRET"] = "whsec_test_secret"
os.environ["SENTRY_DSN"] = "https://test@sentry.io/test"
os.environ["SSL_KEYFILE"] = "/tmp/test.key"
os.environ["SSL_CERTFILE"] = "/tmp/test.cert"

# Import after setting environment variables
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from src.backend.core.auth import get_current_user
from src.backend.database import Base, get_db
from src.backend.main import app
from src.backend.models import User

# Test database URL
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest.fixture(scope="session")
def event_loop() -> Generator:
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="function")
async def test_db():
    """Create a test database for each test function."""
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session() as session:
        yield session

    await engine.dispose()


@pytest.fixture(scope="function")
def override_get_db(test_db):
    """Override the get_db dependency with test database."""

    async def _override_get_db():
        yield test_db

    app.dependency_overrides[get_db] = _override_get_db
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def test_user():
    """Create a test user."""
    return User(
        id=1,
        username="testuser",
        email="test@example.com",
        hashed_password="$2b$12$KIXxPfnK6JpG8KPK8KPK8KPK8KPK8KPK8KPK8KPK8KPK8KPK8KPK8",
        is_active=True,
        is_verified=True,
        role="researcher",
    )


@pytest.fixture
def authenticated_client(test_user, override_get_db):
    """Create a test client with authentication."""

    def override_current_user():
        return test_user

    app.dependency_overrides[get_current_user] = override_current_user

    with TestClient(app) as client:
        yield client

    app.dependency_overrides.clear()


@pytest.fixture
def unauthenticated_client(override_get_db):
    """Create a test client without authentication."""
    with TestClient(app) as client:
        yield client


@pytest.fixture
def mock_adk():
    """Mock ADK client."""
    with patch(
        "nexus_forge.agents.agents.nexus_forge_agents.StarriOrchestrator"
    ) as mock:
        adk_instance = AsyncMock()
        mock.return_value = adk_instance

        # Mock build_app_with_starri method
        async def mock_build_app(*args, **kwargs):
            yield {
                "type": "start",
                "agent": "starri_orchestrator",
                "content": "Starting app build",
                "progress": 0,
            }
            yield {
                "type": "complete",
                "agent": "starri_orchestrator",
                "content": {"summary": "App build complete"},
                "progress": 100,
            }

        adk_instance.build_app_with_starri = mock_build_app

        # Mock health check
        adk_instance.get_agent_health = AsyncMock(
            return_value={
                "overall_status": "healthy",
                "agents": {
                    "orchestrator": {"status": "healthy"},
                    "retrieval": {"status": "healthy"},
                    "analysis": {"status": "healthy"},
                    "citation": {"status": "healthy"},
                    "knowledge_graph": {"status": "healthy"},
                },
            }
        )

        yield adk_instance


@pytest.fixture
def mock_redis():
    """Mock Redis client."""
    with patch("nexus_forge.api.cache.redis_client") as mock:
        redis_instance = AsyncMock()
        mock.return_value = redis_instance

        # Mock basic Redis operations
        redis_instance.get = AsyncMock(return_value=None)
        redis_instance.set = AsyncMock(return_value=True)
        redis_instance.delete = AsyncMock(return_value=1)
        redis_instance.exists = AsyncMock(return_value=0)
        redis_instance.expire = AsyncMock(return_value=True)

        yield redis_instance


@pytest.fixture
def mock_stripe():
    """Mock Stripe client."""
    with (
        patch("stripe.Customer") as customer_mock,
        patch("stripe.Subscription") as subscription_mock,
        patch("stripe.PaymentMethod") as payment_mock,
    ):

        # Mock customer creation
        customer_mock.create = Mock(return_value=Mock(id="cus_test123"))

        # Mock subscription creation
        subscription_mock.create = Mock(
            return_value=Mock(
                id="sub_test123", status="active", current_period_end=1234567890
            )
        )

        # Mock payment method
        payment_mock.attach = Mock()

        yield {
            "customer": customer_mock,
            "subscription": subscription_mock,
            "payment": payment_mock,
        }


@pytest.fixture
def mock_email():
    """Mock email service."""
    with patch("nexus_forge.api.services.email.send_email") as mock:
        mock.return_value = AsyncMock(return_value=True)
        yield mock


@pytest.fixture
def sample_research_query():
    """Sample research query for testing."""
    return {
        "query": "What is quantum computing?",
        "mode": "comprehensive",
        "focus_areas": ["technology", "science"],
        "language": "en",
    }


@pytest.fixture
def sample_websocket_message():
    """Sample WebSocket message for testing."""
    return {
        "type": "research_query",
        "data": {"query": "Test research query", "mode": "quick"},
        "session_id": "550e8400-e29b-41d4-a716-446655440000",
    }


# ====== NEXUS FORGE SPECIFIC FIXTURES ======


@pytest_asyncio.fixture
async def mock_enhanced_starri_orchestrator():
    """Mock enhanced Starri Orchestrator with deep thinking capabilities"""
    from src.backend.agents.starri.orchestrator import (
        AgentCapability,
        StarriOrchestrator,
        ThinkingMode,
    )

    with (
        patch("nexus_forge.agents.starri.orchestrator.GeminiClient"),
        patch("nexus_forge.agents.starri.orchestrator.SupabaseCoordinationClient"),
        patch("nexus_forge.agents.starri.orchestrator.Mem0KnowledgeClient"),
        patch("nexus_forge.agents.starri.orchestrator.RedisCache"),
    ):

        orchestrator = StarriOrchestrator(
            project_id="test-project",
            supabase_url="https://test.supabase.co",
            supabase_key="test-key",
        )

        # Mock all clients
        orchestrator.gemini_client = AsyncMock()
        orchestrator.coordination_client = AsyncMock()
        orchestrator.knowledge_client = AsyncMock()
        orchestrator.cache = AsyncMock()

        # Mock thinking capabilities
        orchestrator.think_deeply = AsyncMock(
            return_value={
                "thinking_chain": [
                    {
                        "step": 1,
                        "thought": "Analyzing the problem...",
                        "confidence": 0.8,
                    },
                    {
                        "step": 2,
                        "thought": "Considering solutions...",
                        "confidence": 0.9,
                    },
                ],
                "conclusion": {
                    "conclusion": "Optimal solution found",
                    "confidence": 0.9,
                },
                "confidence": 0.9,
                "thinking_time": 2.5,
            }
        )

        # Mock task decomposition
        orchestrator.decompose_task = AsyncMock(
            return_value={
                "workflow_id": "test_workflow_123",
                "decomposition": {
                    "subtasks": [
                        {
                            "id": "task_1",
                            "description": "UI Design",
                            "required_capabilities": ["ui_design"],
                        },
                        {
                            "id": "task_2",
                            "description": "Backend API",
                            "required_capabilities": ["code_generation"],
                        },
                    ]
                },
                "confidence": 0.85,
            }
        )

        # Mock agent coordination
        orchestrator.coordinate_agents = AsyncMock(
            return_value={
                "workflow_id": "test_workflow_123",
                "status": "completed",
                "results": {"task_1": "success", "task_2": "success"},
                "metrics": {"tasks_completed": 2, "tasks_failed": 0},
            }
        )

        yield orchestrator


@pytest_asyncio.fixture
async def mock_supabase_coordination():
    """Mock Supabase coordination client"""
    from src.backend.integrations.supabase.coordination_client import (
        SupabaseCoordinationClient,
    )

    with patch("nexus_forge.integrations.supabase.coordination_client.create_client"):
        client = SupabaseCoordinationClient(
            url="https://test.supabase.co", key="test-key", project_id="test-project"
        )

        # Mock Supabase client
        client.client = MagicMock()
        client.client.table.return_value.insert.return_value.execute.return_value.data = [
            {"id": "test_id"}
        ]
        client.client.table.return_value.select.return_value.execute.return_value.data = (
            []
        )
        client.client.table.return_value.update.return_value.eq.return_value.execute.return_value = (
            MagicMock()
        )

        # Mock connection success
        client.connected = True

        yield client


@pytest_asyncio.fixture
async def mock_mem0_knowledge():
    """Mock Mem0 knowledge client"""
    from src.backend.integrations.mem0.knowledge_client import Mem0KnowledgeClient

    with patch("nexus_forge.integrations.mem0.knowledge_client.RedisCache"):
        client = Mem0KnowledgeClient(api_key="test-key", orchestrator_id="test_orch")

        client.cache = MagicMock()
        client.entity_cache = {}
        client.relationship_cache = {}

        # Mock methods
        client.create_entity = AsyncMock(return_value="entity_123")
        client.create_relationship = AsyncMock(return_value="rel_123")
        client.search_patterns = AsyncMock(return_value=[])
        client.add_thinking_pattern = AsyncMock(return_value="pattern_123")

        yield client


@pytest_asyncio.fixture
async def mock_redis_cache():
    """Mock Redis cache with multi-level caching"""
    from src.backend.core.cache import CacheStrategy, RedisCache

    with patch("nexus_forge.core.cache.redis.StrictRedis"):
        cache = RedisCache()
        cache.client = MagicMock()
        cache.client.ping.return_value = True

        # Mock L1 cache
        cache.l1_cache = {}

        # Mock cache operations
        cache.get = MagicMock(return_value=None)
        cache.set = MagicMock(return_value=True)
        cache.get_l1 = MagicMock(return_value=None)
        cache.set_l1 = MagicMock(return_value=True)
        cache.get_l2 = MagicMock(return_value=None)
        cache.set_l2 = MagicMock(return_value=True)
        cache.get_l3 = MagicMock(return_value=None)
        cache.set_l3 = MagicMock(return_value=True)

        # Mock metrics
        cache.get_cache_stats = MagicMock(
            return_value={"hit_rate": 85.5, "total_hits": 100, "total_misses": 15}
        )

        yield cache


@pytest.fixture
def sample_thinking_modes():
    """Sample data for testing thinking modes"""
    from src.backend.agents.starri.orchestrator import ThinkingMode

    return {
        ThinkingMode.DEEP_ANALYSIS: {
            "prompt": "Analyze this complex optimization problem",
            "expected_steps": 3,
            "expected_confidence": 0.8,
        },
        ThinkingMode.QUICK_DECISION: {
            "prompt": "Make a quick decision on framework choice",
            "expected_steps": 2,
            "expected_confidence": 0.7,
        },
        ThinkingMode.PLANNING: {
            "prompt": "Plan the implementation of this feature",
            "expected_steps": 4,
            "expected_confidence": 0.85,
        },
        ThinkingMode.COORDINATION: {
            "prompt": "Coordinate multiple agents for this task",
            "expected_steps": 3,
            "expected_confidence": 0.8,
        },
        ThinkingMode.REFLECTION: {
            "prompt": "Reflect on the completed workflow",
            "expected_steps": 2,
            "expected_confidence": 0.9,
        },
    }


@pytest.fixture
def sample_agent_capabilities():
    """Sample agent capabilities for testing"""
    from src.backend.agents.starri.orchestrator import AgentCapability

    return {
        "ui_designer": [AgentCapability.UI_DESIGN, AgentCapability.IMAGE_GENERATION],
        "backend_developer": [
            AgentCapability.CODE_GENERATION,
            AgentCapability.API_INTEGRATION,
        ],
        "fullstack_developer": [
            AgentCapability.CODE_GENERATION,
            AgentCapability.UI_DESIGN,
            AgentCapability.TESTING,
            AgentCapability.DEPLOYMENT,
        ],
        "qa_engineer": [AgentCapability.TESTING, AgentCapability.DATA_ANALYSIS],
        "devops_engineer": [AgentCapability.DEPLOYMENT, AgentCapability.OPTIMIZATION],
    }


@pytest.fixture
def sample_workflow_decomposition():
    """Sample workflow decomposition for testing"""
    return {
        "workflow_id": "sample_workflow_001",
        "decomposition": {
            "subtasks": [
                {
                    "id": "design_ui",
                    "description": "Design user interface components",
                    "required_capabilities": ["ui_design"],
                    "estimated_duration": "60m",
                    "dependencies": [],
                    "complexity": "medium",
                },
                {
                    "id": "implement_backend",
                    "description": "Create REST API endpoints",
                    "required_capabilities": ["code_generation", "api_integration"],
                    "estimated_duration": "90m",
                    "dependencies": [],
                    "complexity": "medium",
                },
                {
                    "id": "integrate_frontend",
                    "description": "Connect frontend to backend",
                    "required_capabilities": ["code_generation"],
                    "estimated_duration": "45m",
                    "dependencies": ["design_ui", "implement_backend"],
                    "complexity": "low",
                },
                {
                    "id": "write_tests",
                    "description": "Create comprehensive test suite",
                    "required_capabilities": ["testing"],
                    "estimated_duration": "60m",
                    "dependencies": ["integrate_frontend"],
                    "complexity": "medium",
                },
                {
                    "id": "deploy_application",
                    "description": "Deploy to production environment",
                    "required_capabilities": ["deployment"],
                    "estimated_duration": "30m",
                    "dependencies": ["write_tests"],
                    "complexity": "low",
                },
            ],
            "total_duration": "285m",
            "critical_path": [
                "design_ui",
                "integrate_frontend",
                "write_tests",
                "deploy_application",
            ],
            "execution_strategy": "mixed",
        },
        "confidence": 0.88,
    }


@pytest.fixture
def performance_thresholds():
    """Performance thresholds for testing"""
    return {
        "thinking_time": {"max_per_step": 5.0, "max_total": 30.0},  # seconds  # seconds
        "task_decomposition": {"max_time": 10.0, "min_confidence": 0.7},  # seconds
        "agent_coordination": {
            "max_setup_time": 5.0,  # seconds
            "max_execution_time": 300.0,  # 5 minutes
            "min_success_rate": 0.9,
        },
        "cache_operations": {
            "l1_max_time": 0.001,  # 1ms
            "l2_max_time": 0.01,  # 10ms
            "l3_max_time": 0.1,  # 100ms
        },
    }


# ====== NEXUS FORGE SPECIFIC FIXTURES ======


@pytest_asyncio.fixture
async def mock_starri_orchestrator():
    """Mock Starri Orchestrator for testing"""
    with patch(
        "nexus_forge.agents.agents.nexus_forge_agents.StarriOrchestrator"
    ) as mock_class:
        orchestrator = AsyncMock()
        mock_class.return_value = orchestrator

        # Mock core methods
        orchestrator.build_app_with_starri = AsyncMock(
            return_value={
                "specification": {"name": "Test App", "description": "Test"},
                "mockups": {"Dashboard": "https://mockup.url"},
                "demo_video": "https://video.url",
                "code_files": {"main.py": "# Generated code"},
                "deployment_url": "https://app.run.app",
                "build_time": "3 minutes 45 seconds",
                "orchestrator": "Starri",
                "models_used": ["gemini_pro", "imagen", "veo", "jules", "gemini_flash"],
            }
        )

        orchestrator.generate_app_specification = AsyncMock()
        orchestrator.generate_ui_mockups = AsyncMock()
        orchestrator.generate_demo_video = AsyncMock()
        orchestrator.generate_code_with_jules = AsyncMock()

        yield orchestrator


@pytest_asyncio.fixture
async def mock_veo_integration():
    """Mock Veo 3 integration for testing"""
    with patch("nexus_forge.integrations.veo_integration.VeoIntegration") as mock_class:
        veo = AsyncMock()
        mock_class.return_value = veo

        veo.generate_demo_video = AsyncMock(
            return_value="https://storage.googleapis.com/test-videos/demo.mp4"
        )
        veo.generate_user_flow_animation = AsyncMock(
            return_value="https://storage.googleapis.com/test-videos/flow.mp4"
        )
        veo.generate_feature_showcase = AsyncMock(
            return_value="https://storage.googleapis.com/test-videos/feature.mp4"
        )

        yield veo


@pytest_asyncio.fixture
async def mock_imagen_integration():
    """Mock Imagen 4 integration for testing"""
    with patch(
        "nexus_forge.integrations.imagen_integration.ImagenIntegration"
    ) as mock_class:
        imagen = AsyncMock()
        mock_class.return_value = imagen

        imagen.generate_ui_mockup = AsyncMock(
            return_value={
                "url": "https://storage.googleapis.com/test-mockups/component.png",
                "metadata": {"component": "Dashboard", "style": "modern"},
                "format": "png",
                "resolution": "2K",
            }
        )

        imagen.generate_design_system = AsyncMock(
            return_value={
                "colors": {
                    "primary": "#3B82F6",
                    "secondary": "#8B5CF6",
                    "neutral": {"500": "#6B7280"},
                },
                "typography": {
                    "fontFamily": {"sans": "Inter"},
                    "fontSize": {"base": "1rem"},
                },
                "components": {},
                "spacing": {"4": "1rem"},
                "shadows": {"md": "0 4px 6px rgba(0,0,0,0.1)"},
            }
        )

        imagen.generate_page_layout = AsyncMock(
            return_value="https://storage.googleapis.com/test-mockups/layout.png"
        )

        yield imagen


@pytest_asyncio.fixture
async def mock_nexus_forge_websocket():
    """Mock WebSocket for Nexus Forge real-time updates"""
    websocket = AsyncMock()
    websocket.accept = AsyncMock()
    websocket.send_json = AsyncMock()
    websocket.receive_json = AsyncMock()
    websocket.close = AsyncMock()

    # Configure for build request simulation
    websocket.receive_json.return_value = {
        "type": "build_request",
        "data": {"prompt": "Build a test app", "config": {"useAdaptiveThinking": True}},
    }

    yield websocket


@pytest.fixture
def sample_app_specification():
    """Sample app specification for testing"""
    from src.backend.agents.agents.nexus_forge_agents import AppSpecification

    return AppSpecification(
        name="Test Analytics App",
        description="A test analytics dashboard application",
        features=[
            "Real-time data visualization",
            "User authentication",
            "Export functionality",
            "Interactive charts",
        ],
        tech_stack={
            "frontend": "React",
            "backend": "FastAPI",
            "database": "PostgreSQL",
        },
        ui_components=["Dashboard", "Charts", "UserProfile", "Settings"],
        api_endpoints=[
            {
                "method": "GET",
                "path": "/api/data",
                "description": "Fetch analytics data",
            },
            {
                "method": "POST",
                "path": "/api/auth",
                "description": "User authentication",
            },
        ],
        database_schema={"tables": ["users", "analytics_data", "sessions"]},
        deployment_config={
            "platform": "Cloud Run",
            "scaling": "auto",
            "environment": "production",
        },
    )


@pytest.fixture
def sample_build_session():
    """Sample build session data for testing"""
    return {
        "session_id": "test-session-123",
        "user_id": 1,
        "prompt": "Build a real-time analytics dashboard",
        "status": "building",
        "started_at": "2024-01-15T10:00:00Z",
        "config": {
            "useAdaptiveThinking": True,
            "enableVideoDemo": True,
            "deployToCloudRun": True,
        },
    }


@pytest_asyncio.fixture
async def mock_ai_model_responses():
    """Mock responses from all AI models"""
    return {
        "gemini_2_5_pro": {
            "specification": {
                "name": "Generated App",
                "description": "AI-generated application",
                "features": ["Feature 1", "Feature 2"],
                "tech_stack": {"frontend": "React", "backend": "FastAPI"},
                "ui_components": ["Dashboard", "Settings"],
                "api_endpoints": [{"method": "GET", "path": "/api/health"}],
                "database_schema": {"tables": ["users"]},
                "deployment_config": {"platform": "Cloud Run"},
            }
        },
        "imagen_4": {
            "mockups": {
                "Dashboard": "https://mockups.test/dashboard.png",
                "Settings": "https://mockups.test/settings.png",
            },
            "design_system": {
                "colors": {"primary": "#007bff"},
                "typography": {"base": "16px"},
            },
        },
        "veo_3": {
            "demo_video": "https://videos.test/demo.mp4",
            "feature_videos": {"Feature 1": "https://videos.test/feature1.mp4"},
        },
        "jules": {
            "code_files": {
                "main.py": "# FastAPI application\nfrom fastapi import FastAPI\napp = FastAPI()",
                "frontend/App.tsx": "// React application\nimport React from 'react';",
                "tests/test_main.py": "# Tests\nimport pytest",
            }
        },
        "gemini_flash": {
            "optimized_code": {
                "main.py": "# Optimized FastAPI application",
                "frontend/App.tsx": "// Optimized React application",
            }
        },
    }


@pytest.fixture
def performance_benchmarks():
    """Performance benchmarks for testing"""
    return {
        "api_response_time": 3.0,  # seconds
        "websocket_connection_time": 0.5,  # seconds
        "build_completion_time": 600.0,  # 10 minutes max
        "memory_usage_limit": 1024 * 1024 * 1024,  # 1GB
        "concurrent_user_limit": 100,
    }


@pytest_asyncio.fixture
async def integration_test_environment():
    """Set up environment for integration tests"""
    # Mock external service endpoints
    mock_services = {
        "vertex_ai_endpoint": "https://mock-vertex-ai.googleapis.com",
        "imagen_endpoint": "https://mock-imagen.googleapis.com",
        "veo_endpoint": "https://mock-veo.googleapis.com",
        "cloud_run_endpoint": "https://mock-cloudrun.googleapis.com",
    }

    # Set environment variables for testing
    original_env = os.environ.copy()
    os.environ.update(
        {
            "VERTEX_AI_ENDPOINT": mock_services["vertex_ai_endpoint"],
            "IMAGEN_ENDPOINT": mock_services["imagen_endpoint"],
            "VEO_ENDPOINT": mock_services["veo_endpoint"],
            "CLOUD_RUN_ENDPOINT": mock_services["cloud_run_endpoint"],
            "INTEGRATION_TEST_MODE": "true",
        }
    )

    yield mock_services

    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers for Nexus Forge"""
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "security: mark test as security test")
    config.addinivalue_line("markers", "performance: mark test as performance test")
    config.addinivalue_line("markers", "nexus_forge: mark test as Nexus Forge specific")
    config.addinivalue_line(
        "markers", "multi_agent: mark test as multi-agent orchestration test"
    )
    config.addinivalue_line("markers", "websocket: mark test as WebSocket related")


# Test utilities for Nexus Forge
class MockAIModel:
    """Utility class for mocking AI model responses"""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.call_count = 0
        self.responses = []

    async def generate_content_async(self, prompt: str, **kwargs):
        """Mock content generation"""
        self.call_count += 1
        response = MagicMock()
        response.text = f"Generated content for {self.model_name}: {prompt[:50]}..."
        return response

    def add_response(self, response_text: str):
        """Add a predefined response"""
        self.responses.append(response_text)


async def simulate_build_progress(websocket: AsyncMock, phases: list):
    """Simulate build progress updates via WebSocket"""
    for i, phase in enumerate(phases):
        progress = int((i + 1) / len(phases) * 100)
        await websocket.send_json(
            {
                "type": "progress_update",
                "phase": phase["id"],
                "message": f"Processing {phase['name']}...",
                "progress": progress,
                "model": phase["model"],
            }
        )
        await asyncio.sleep(0.1)  # Simulate processing time
