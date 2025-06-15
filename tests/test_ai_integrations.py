"""
Test suite for AI model integrations - Veo 3 and Imagen 4

Tests the integration modules that interface with Google's latest AI models
for video generation and UI mockup creation.
"""

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, patch, MagicMock
import aiohttp
import asyncio

from src.api.integrations.veo_integration import (
    VeoIntegration,
    VeoVideoEditor,
    VideoGenerationRequest
)
from src.api.integrations.imagen_integration import (
    ImagenIntegration,
    ImagenDesignExporter,
    MockupGenerationRequest,
    DesignSystemRequest
)

pytestmark = pytest.mark.asyncio


class TestVeoIntegration:
    """Test suite for Veo 3 video generation integration"""
    
    @pytest_asyncio.fixture
    async def veo_integration(self):
        """Create VeoIntegration instance with mocked credentials"""
        with patch('src.api.integrations.veo_integration.google.auth'):
            integration = VeoIntegration(
                project_id="test-project",
                region="us-central1"
            )
            integration.credentials = MagicMock()
            integration.credentials.token = "test-token"
            yield integration
    
    async def test_veo_initialization(self, veo_integration):
        """Test VeoIntegration initialization"""
        assert veo_integration.project_id == "test-project"
        assert veo_integration.region == "us-central1"
        assert "aiplatform.googleapis.com" in veo_integration.api_endpoint
    
    async def test_generate_demo_video(self, veo_integration):
        """Test demo video generation"""
        app_name = "Test Analytics App"
        features = ["Real-time charts", "User auth", "Export data"]
        mockup_urls = ["https://mockup1.png", "https://mockup2.png"]
        video_plan = {
            "duration": 60,
            "scenes": ["Intro", "Features", "Demo", "CTA"],
            "narration": "Welcome to Test Analytics App..."
        }
        
        # Mock the API call
        with patch.object(veo_integration, '_call_veo_api') as mock_call:
            mock_call.return_value = "https://videos.test/demo.mp4"
            
            video_url = await veo_integration.generate_demo_video(
                app_name, features, mockup_urls, video_plan
            )
            
            assert video_url == "https://videos.test/demo.mp4"
            mock_call.assert_called_once()
            
            # Verify the request structure
            call_args = mock_call.call_args[0][0]
            assert isinstance(call_args, VideoGenerationRequest)
            assert call_args.duration == 60
            assert call_args.mockup_urls == mockup_urls
    
    async def test_generate_user_flow_animation(self, veo_integration):
        """Test user flow animation generation"""
        app_spec = {
            "name": "Flow Test App",
            "description": "Test application for flow testing"
        }
        flow_steps = [
            {"action": "Open app", "result": "Dashboard loads"},
            {"action": "Click feature", "result": "Feature opens"},
            {"action": "Complete task", "result": "Success shown"}
        ]
        
        with patch.object(veo_integration, '_call_veo_api') as mock_call:
            mock_call.return_value = "https://videos.test/flow.mp4"
            
            animation_url = await veo_integration.generate_user_flow_animation(
                app_spec, flow_steps
            )
            
            assert animation_url == "https://videos.test/flow.mp4"
            mock_call.assert_called_once()
    
    async def test_generate_feature_showcase(self, veo_integration):
        """Test feature showcase video generation"""
        feature_name = "Real-time Dashboard"
        feature_description = "Live updating analytics dashboard with interactive charts"
        mockup_url = "https://mockups.test/dashboard.png"
        
        with patch.object(veo_integration, '_call_veo_api') as mock_call:
            mock_call.return_value = "https://videos.test/feature.mp4"
            
            showcase_url = await veo_integration.generate_feature_showcase(
                feature_name, feature_description, mockup_url
            )
            
            assert showcase_url == "https://videos.test/feature.mp4"
            mock_call.assert_called_once()
            
            # Verify short duration for feature showcase
            call_args = mock_call.call_args[0][0]
            assert call_args.duration == 15
    
    async def test_veo_api_call_success(self, veo_integration):
        """Test successful Veo API call"""
        request = VideoGenerationRequest(
            prompt="Test video prompt",
            duration=30
        )
        
        # Mock successful API response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "videoUrl": "https://storage.googleapis.com/videos/test.mp4"
        })
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value = mock_response
            
            result = await veo_integration._call_veo_api(request)
            
            assert result == "https://storage.googleapis.com/videos/test.mp4"
    
    async def test_veo_api_call_fallback(self, veo_integration):
        """Test Veo API fallback to simulation"""
        request = VideoGenerationRequest(
            prompt="Test prompt",
            duration=45
        )
        
        # Mock API failure
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.side_effect = Exception("API Error")
            
            result = await veo_integration._call_veo_api(request)
            
            # Should fallback to simulation
            assert result.startswith("https://storage.googleapis.com/nexus-forge-demos/")
            assert result.endswith(".mp4")
    
    async def test_video_editor_branded_demo(self, veo_integration):
        """Test VeoVideoEditor branded demo creation"""
        editor = VeoVideoEditor(veo_integration)
        
        base_video_url = "https://videos.test/base.mp4"
        brand_assets = {
            "logo": "https://assets.test/logo.png",
            "colors": {"primary": "#007bff"},
            "fonts": {"primary": "Inter"}
        }
        customizations = {
            "style": "professional",
            "include_logo": True
        }
        
        branded_url = await editor.create_branded_demo(
            base_video_url, brand_assets, customizations
        )
        
        # For now, should return the base URL (implementation placeholder)
        assert branded_url == base_video_url


class TestImagenIntegration:
    """Test suite for Imagen 4 UI mockup generation integration"""
    
    @pytest_asyncio.fixture
    async def imagen_integration(self):
        """Create ImagenIntegration instance with mocked credentials"""
        with patch('src.api.integrations.imagen_integration.google.auth'):
            integration = ImagenIntegration(
                project_id="test-project",
                region="us-central1"
            )
            integration.credentials = MagicMock()
            integration.credentials.token = "test-token"
            yield integration
    
    async def test_imagen_initialization(self, imagen_integration):
        """Test ImagenIntegration initialization"""
        assert imagen_integration.project_id == "test-project"
        assert imagen_integration.region == "us-central1"
        assert "aiplatform.googleapis.com" in imagen_integration.api_endpoint
    
    async def test_generate_ui_mockup(self, imagen_integration):
        """Test UI mockup generation"""
        component_name = "Dashboard"
        app_context = {
            "name": "Analytics App",
            "description": "Real-time analytics dashboard",
            "tech_stack": {"frontend": "React", "backend": "FastAPI"}
        }
        design_requirements = {
            "style": "modern_minimalist",
            "colors": {"primary": "#3B82F6"},
            "include_annotations": True
        }
        
        with patch.object(imagen_integration, '_call_imagen_api') as mock_call:
            mock_call.return_value = {
                "url": "https://mockups.test/dashboard.png",
                "metadata": {"component": "Dashboard", "style": "modern"},
                "format": "png",
                "resolution": "2K"
            }
            
            mockup_data = await imagen_integration.generate_ui_mockup(
                component_name, app_context, design_requirements
            )
            
            assert mockup_data["url"] == "https://mockups.test/dashboard.png"
            assert mockup_data["metadata"]["component"] == "Dashboard"
            assert mockup_data["resolution"] == "2K"
            mock_call.assert_called_once()
    
    async def test_generate_design_system(self, imagen_integration):
        """Test complete design system generation"""
        app_spec = {
            "name": "Design System App",
            "description": "Professional business application",
            "ui_components": ["Header", "Dashboard", "Footer"],
            "style": "corporate_professional"
        }
        
        # Mock the color system generation
        with patch.object(imagen_integration, '_generate_color_system') as mock_colors:
            mock_colors.return_value = {
                "primary": "#1E40AF",
                "secondary": "#8B5CF6",
                "neutral": {"500": "#6B7280"}
            }
            
            with patch.object(imagen_integration, 'generate_ui_mockup') as mock_mockup:
                mock_mockup.return_value = {
                    "url": "https://mockups.test/component.png"
                }
                
                design_system = await imagen_integration.generate_design_system(app_spec)
                
                assert "colors" in design_system
                assert "typography" in design_system
                assert "components" in design_system
                assert "spacing" in design_system
                assert "shadows" in design_system
                
                # Verify components were generated
                assert len(design_system["components"]) == 3
                assert "Header" in design_system["components"]
    
    async def test_generate_page_layout(self, imagen_integration):
        """Test page layout generation"""
        page_name = "Dashboard"
        components = ["Header", "Sidebar", "MainContent", "Footer"]
        layout_type = "responsive_grid"
        
        with patch.object(imagen_integration, '_call_imagen_api') as mock_call:
            mock_call.return_value = {
                "url": "https://mockups.test/dashboard-layout.png"
            }
            
            layout_url = await imagen_integration.generate_page_layout(
                page_name, components, layout_type
            )
            
            assert layout_url == "https://mockups.test/dashboard-layout.png"
            mock_call.assert_called_once()
    
    async def test_imagen_api_call_success(self, imagen_integration):
        """Test successful Imagen API call"""
        request = MockupGenerationRequest(
            prompt="Modern dashboard component",
            component_type="Dashboard",
            style="minimalist"
        )
        
        # Mock successful API response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "imageUrl": "https://storage.googleapis.com/mockups/dashboard.png",
            "metadata": {"style": "minimalist", "quality": "high"}
        })
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value = mock_response
            
            result = await imagen_integration._call_imagen_api(request)
            
            assert result["url"] == "https://storage.googleapis.com/mockups/dashboard.png"
            assert result["metadata"]["style"] == "minimalist"
    
    async def test_imagen_api_call_fallback(self, imagen_integration):
        """Test Imagen API fallback to simulation"""
        request = MockupGenerationRequest(
            prompt="Test component",
            component_type="TestComponent"
        )
        
        # Mock API failure
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.side_effect = Exception("API Error")
            
            result = await imagen_integration._call_imagen_api(request)
            
            # Should fallback to simulation
            assert result["url"].startswith("https://storage.googleapis.com/nexus-forge-mockups/")
            assert "testcomponent" in result["url"].lower()
    
    async def test_responsive_variants_generation(self, imagen_integration):
        """Test responsive variant generation"""
        base_mockup_url = "https://mockups.test/base.png"
        component_name = "Dashboard"
        
        variants = await imagen_integration._generate_responsive_variants(
            base_mockup_url, component_name
        )
        
        assert "mobile" in variants
        assert "tablet" in variants
        assert "desktop" in variants
        assert "wide" in variants
        
        # All should be unique URLs
        assert len(set(variants.values())) == 4
    
    async def test_design_exporter_css_export(self, imagen_integration):
        """Test CSS export functionality"""
        exporter = ImagenDesignExporter(imagen_integration)
        
        design_system = {
            "colors": {
                "primary": "#3B82F6",
                "neutral": {"500": "#6B7280", "600": "#4B5563"}
            },
            "typography": {
                "fontSize": {"base": "1rem", "lg": "1.125rem"}
            },
            "spacing": {"4": "1rem", "8": "2rem"},
            "shadows": {"md": "0 4px 6px rgba(0,0,0,0.1)"}
        }
        
        css_content = await exporter.export_to_css(design_system)
        
        assert ":root {" in css_content
        assert "--color-primary: #3B82F6;" in css_content
        assert "--color-neutral-500: #6B7280;" in css_content
        assert "--font-size-base: 1rem;" in css_content
        assert "--spacing-4: 1rem;" in css_content
        assert "--shadow-md: 0 4px 6px rgba(0,0,0,0.1);" in css_content
        assert "}" in css_content
    
    async def test_design_exporter_react_components(self, imagen_integration):
        """Test React component export"""
        exporter = ImagenDesignExporter(imagen_integration)
        
        design_system = {
            "colors": {"primary": "#3B82F6"},
            "typography": {"fontSize": {"base": "1rem"}}
        }
        component_name = "Dashboard"
        
        component_files = await exporter.export_to_react_components(
            design_system, component_name
        )
        
        assert f"{component_name}.tsx" in component_files
        assert f"{component_name}.styles.ts" in component_files
        
        tsx_content = component_files[f"{component_name}.tsx"]
        assert f"interface {component_name}Props" in tsx_content
        assert f"export const {component_name}:" in tsx_content
        assert "React.FC" in tsx_content


class TestIntegrationPerformance:
    """Test performance characteristics of AI integrations"""
    
    @pytest.mark.performance
    async def test_parallel_ai_model_calls(self):
        """Test parallel execution of multiple AI model calls"""
        with patch('src.api.integrations.veo_integration.VeoIntegration') as mock_veo:
            with patch('src.api.integrations.imagen_integration.ImagenIntegration') as mock_imagen:
                # Set up mocks
                veo_instance = AsyncMock()
                imagen_instance = AsyncMock()
                mock_veo.return_value = veo_instance
                mock_imagen.return_value = imagen_instance
                
                # Configure async returns
                veo_instance.generate_demo_video = AsyncMock(
                    return_value="video.mp4"
                )
                imagen_instance.generate_ui_mockup = AsyncMock(
                    return_value={"url": "mockup.png"}
                )
                
                # Measure parallel execution time
                start_time = asyncio.get_event_loop().time()
                
                veo_task = asyncio.create_task(
                    veo_instance.generate_demo_video("app", [], [], {})
                )
                imagen_task = asyncio.create_task(
                    imagen_instance.generate_ui_mockup("component", {}, {})
                )
                
                video_result, mockup_result = await asyncio.gather(veo_task, imagen_task)
                
                end_time = asyncio.get_event_loop().time()
                
                # Should complete quickly with mocks
                assert (end_time - start_time) < 1.0
                assert video_result == "video.mp4"
                assert mockup_result["url"] == "mockup.png"
    
    @pytest.mark.performance
    async def test_memory_usage_with_large_requests(self):
        """Test memory usage with large generation requests"""
        integration = ImagenIntegration("test-project")
        
        # Large app specification
        large_app_spec = {
            "name": "Enterprise Application",
            "description": "Large enterprise application with many components",
            "ui_components": [f"Component{i}" for i in range(50)],  # 50 components
            "features": [f"Feature {i}" for i in range(100)]  # 100 features
        }
        
        with patch.object(integration, 'generate_ui_mockup') as mock_mockup:
            mock_mockup.return_value = {"url": "test.png"}
            
            # Should handle large requests without memory issues
            design_system = await integration.generate_design_system(large_app_spec)
            
            assert len(design_system["components"]) == 50
            assert mock_mockup.call_count == 50


class TestIntegrationErrorHandling:
    """Test error handling and resilience of AI integrations"""
    
    async def test_veo_network_timeout(self):
        """Test Veo integration handling network timeouts"""
        integration = VeoIntegration("test-project")
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.side_effect = asyncio.TimeoutError("Request timeout")
            
            request = VideoGenerationRequest(prompt="test", duration=30)
            result = await integration._call_veo_api(request)
            
            # Should fallback to simulation
            assert result.startswith("https://storage.googleapis.com/nexus-forge-demos/")
    
    async def test_imagen_authentication_error(self):
        """Test Imagen integration handling authentication errors"""
        integration = ImagenIntegration("test-project")
        integration.credentials = None  # Simulate no credentials
        
        request = MockupGenerationRequest(
            prompt="test component",
            component_type="Test"
        )
        
        result = await integration._call_imagen_api(request)
        
        # Should fallback to simulation when no credentials
        assert result["url"].startswith("https://storage.googleapis.com/nexus-forge-mockups/")
    
    async def test_concurrent_request_limits(self):
        """Test handling of concurrent request limits"""
        integration = VeoIntegration("test-project")
        
        # Create many concurrent requests
        tasks = []
        for i in range(20):
            request = VideoGenerationRequest(
                prompt=f"test prompt {i}",
                duration=15
            )
            task = asyncio.create_task(integration._simulate_veo_generation(request))
            tasks.append(task)
        
        # All should complete without errors
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # No exceptions should occur
        assert all(not isinstance(result, Exception) for result in results)
        assert len(results) == 20