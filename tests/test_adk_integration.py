"""
Comprehensive test suite for ADK integration

Tests the native ADK implementation, agent coordination,
and multi-agent research capabilities.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime
import json

from src.api.adk_integration import ParallaxPalADK
from google.adk.agents import LlmAgent
from google.adk.tools import GoogleSearchTool, CodeExecTool


class TestADKIntegration:
    """Test suite for ADK integration"""
    
    @pytest.fixture
    async def adk_client(self):
        """Create ADK client for testing"""
        with patch.dict('os.environ', {
            'GOOGLE_CLOUD_PROJECT': 'test-project',
            'GOOGLE_CLOUD_LOCATION': 'us-central1'
        }):
            # Mock vertexai initialization
            with patch('src.api.adk_integration.vertexai.init'):
                client = ParallaxPalADK()
                yield client
    
    @pytest.fixture
    def mock_streaming_session(self):
        """Mock streaming session for testing"""
        session = AsyncMock()
        
        async def mock_stream_query(query):
            """Generate mock streaming events"""
            events = [
                Mock(type='start', agent_name='orchestrator', content='Starting research', 
                     progress=5, metadata={}),
                Mock(type='delegating', agent_name='orchestrator', content='Delegating to retrieval', 
                     progress=10, metadata={}),
                Mock(type='searching', agent_name='retrieval_agent', content='Searching sources', 
                     progress=30, metadata={}),
                Mock(type='analyzing', agent_name='analysis_agent', content='Analyzing data', 
                     progress=50, metadata={}),
                Mock(type='citing', agent_name='citation_agent', content='Generating citations', 
                     progress=70, metadata={}),
                Mock(type='graphing', agent_name='knowledge_graph_agent', content='Building graph', 
                     progress=85, metadata={}),
                Mock(type='complete', agent_name='orchestrator', content={'summary': 'Test complete'}, 
                     progress=100, metadata={})
            ]
            
            for event in events:
                yield event
        
        session.stream_query = mock_stream_query
        return session
    
    @pytest.mark.asyncio
    async def test_agent_initialization(self, adk_client):
        """Test all agents are properly initialized"""
        
        # Check all required agents exist
        assert 'orchestrator' in adk_client.agents
        assert 'retrieval' in adk_client.agents
        assert 'analysis' in adk_client.agents
        assert 'citation' in adk_client.agents
        assert 'knowledge_graph' in adk_client.agents
        
        # Verify agent types
        for agent_name, agent in adk_client.agents.items():
            assert isinstance(agent, LlmAgent)
        
        # Verify orchestrator has sub-agents
        orchestrator = adk_client.agents['orchestrator']
        assert hasattr(orchestrator, 'sub_agents')
        assert len(orchestrator.sub_agents) == 4
        
        # Verify retrieval agent has Google Search tool
        retrieval = adk_client.agents['retrieval']
        assert any(isinstance(tool, GoogleSearchTool) for tool in retrieval.tools)
        
        # Verify analysis agent has Code Execution tool
        analysis = adk_client.agents['analysis']
        assert any(isinstance(tool, CodeExecTool) for tool in analysis.tools)
    
    @pytest.mark.asyncio
    async def test_stream_research_basic(self, adk_client, mock_streaming_session):
        """Test basic research streaming functionality"""
        
        with patch('src.api.adk_integration.StreamingSession', return_value=mock_streaming_session):
            events = []
            
            async for event in adk_client.stream_research(
                "What is quantum computing?",
                "user123",
                "session456",
                "comprehensive"
            ):
                events.append(event)
            
            # Verify event structure
            assert len(events) == 7
            assert events[0]['type'] == 'start'
            assert events[-1]['type'] == 'complete'
            assert events[-1]['progress'] == 100
            
            # Verify all agents participated
            agent_names = {event['agent'] for event in events}
            assert 'orchestrator' in agent_names
            assert 'retrieval_agent' in agent_names
            assert 'analysis_agent' in agent_names
    
    @pytest.mark.asyncio
    async def test_stream_research_error_handling(self, adk_client):
        """Test error handling in research streaming"""
        
        # Mock session that raises an error
        error_session = AsyncMock()
        error_session.stream_query.side_effect = Exception("Test error")
        
        with patch('src.api.adk_integration.StreamingSession', return_value=error_session):
            events = []
            
            async for event in adk_client.stream_research(
                "Test query",
                "user123",
                "session456"
            ):
                events.append(event)
            
            # Should get error event
            assert len(events) == 1
            assert events[0]['type'] == 'error'
            assert events[0]['progress'] == 100
            assert 'orchestrator' in events[0]['agent']
    
    @pytest.mark.asyncio
    async def test_query_preparation(self, adk_client):
        """Test query preparation with different modes"""
        
        # Test quick mode
        quick_query = adk_client._prepare_query("Test query", "quick")
        assert "quick overview" in quick_query
        assert "3-5 key sources" in quick_query
        
        # Test comprehensive mode
        comp_query = adk_client._prepare_query("Test query", "comprehensive")
        assert "thorough research" in comp_query
        assert "10-15 sources" in comp_query
        
        # Test continuous mode
        cont_query = adk_client._prepare_query("Test query", "continuous")
        assert "exhaustively" in cont_query
        assert "alternative viewpoints" in cont_query
    
    @pytest.mark.asyncio
    async def test_progress_calculation(self, adk_client):
        """Test progress calculation for different event types"""
        
        test_cases = [
            (Mock(type='start', agent_name='orchestrator'), 5),
            (Mock(type='searching', agent_name='retrieval_agent'), 30),
            (Mock(type='analyzing', agent_name='analysis_agent'), 50),
            (Mock(type='complete', agent_name='orchestrator'), 100),
            (Mock(type='unknown', agent_name='unknown'), 50)  # Default
        ]
        
        for event, expected_progress in test_cases:
            progress = adk_client._calculate_progress(event)
            assert progress == expected_progress
    
    @pytest.mark.asyncio
    async def test_agent_health_check(self, adk_client):
        """Test agent health monitoring"""
        
        # Mock agent responses
        for agent_name, agent in adk_client.agents.items():
            agent.aquery = AsyncMock(return_value="OK")
        
        health = await adk_client.get_agent_health()
        
        # Verify health structure
        assert 'overall_status' in health
        assert 'agents' in health
        assert 'timestamp' in health
        
        # All agents should be healthy
        assert health['overall_status'] == 'healthy'
        for agent_name in adk_client.agents:
            assert agent_name in health['agents']
            assert health['agents'][agent_name]['status'] == 'healthy'
            assert 'response_time_seconds' in health['agents'][agent_name]
    
    @pytest.mark.asyncio
    async def test_agent_health_check_failure(self, adk_client):
        """Test agent health check with failures"""
        
        # Make one agent unhealthy
        adk_client.agents['orchestrator'].aquery = AsyncMock(return_value="OK")
        adk_client.agents['retrieval'].aquery = AsyncMock(side_effect=Exception("Agent error"))
        adk_client.agents['analysis'].aquery = AsyncMock(return_value="OK")
        adk_client.agents['citation'].aquery = AsyncMock(return_value="OK")
        adk_client.agents['knowledge_graph'].aquery = AsyncMock(return_value="OK")
        
        health = await adk_client.get_agent_health()
        
        # Overall status should be degraded
        assert health['overall_status'] == 'degraded'
        assert health['agents']['retrieval']['status'] == 'unhealthy'
        assert 'error' in health['agents']['retrieval']
    
    @pytest.mark.asyncio
    async def test_multi_agent_coordination(self, adk_client):
        """Test multi-agent coordination flow"""
        
        # Create a more complex mock session
        coordination_session = AsyncMock()
        
        async def mock_coordinated_stream(query):
            """Simulate coordinated multi-agent work"""
            # Orchestrator delegates
            yield Mock(
                type='delegating', 
                agent_name='orchestrator',
                content='Breaking down query into subtasks',
                progress=10,
                metadata={'subtasks': 3}
            )
            
            # Parallel agent work
            yield Mock(
                type='searching',
                agent_name='retrieval_agent',
                content='Found 15 relevant sources',
                progress=30,
                metadata={'source_count': 15}
            )
            
            yield Mock(
                type='analyzing',
                agent_name='analysis_agent',
                content='Synthesizing information',
                progress=50,
                metadata={'patterns_found': 5}
            )
            
            # Sequential refinement
            yield Mock(
                type='citing',
                agent_name='citation_agent',
                content='Generated 15 citations',
                progress=70,
                metadata={'citation_count': 15}
            )
            
            yield Mock(
                type='graphing',
                agent_name='knowledge_graph_agent',
                content='Created knowledge graph',
                progress=85,
                metadata={'node_count': 25, 'edge_count': 40}
            )
            
            # Final synthesis
            yield Mock(
                type='synthesizing',
                agent_name='orchestrator',
                content='Compiling final report',
                progress=95,
                metadata={}
            )
            
            yield Mock(
                type='complete',
                agent_name='orchestrator',
                content={
                    'summary': 'Research complete',
                    'sources': [{'title': 'Source 1', 'url': 'http://example.com'}],
                    'findings': ['Finding 1', 'Finding 2'],
                    'knowledge_graph': {
                        'nodes': [{'id': '1', 'label': 'Concept', 'type': 'concept'}],
                        'edges': [{'source': '1', 'target': '2', 'type': 'relates_to'}]
                    }
                },
                progress=100,
                metadata={'total_time': 45.2}
            )
        
        coordination_session.stream_query = mock_coordinated_stream
        
        with patch('src.api.adk_integration.StreamingSession', return_value=coordination_session):
            events = []
            
            async for event in adk_client.stream_research(
                "Complex research query",
                "user123",
                "session456"
            ):
                events.append(event)
            
            # Verify coordination flow
            assert len(events) == 7
            
            # Check metadata propagation
            assert events[0]['metadata']['subtasks'] == 3
            assert events[1]['metadata']['source_count'] == 15
            assert events[4]['metadata']['node_count'] == 25
            
            # Verify final results
            final_event = events[-1]
            assert final_event['type'] == 'complete'
            assert 'summary' in final_event['content']
            assert 'sources' in final_event['content']
            assert 'knowledge_graph' in final_event['content']


class TestADKErrorScenarios:
    """Test error scenarios and edge cases"""
    
    @pytest.mark.asyncio
    async def test_missing_environment_variables(self):
        """Test handling of missing environment variables"""
        
        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(ValueError, match="GOOGLE_CLOUD_PROJECT"):
                ParallaxPalADK()
    
    @pytest.mark.asyncio
    async def test_invalid_query_mode(self):
        """Test handling of invalid query modes"""
        
        with patch.dict('os.environ', {
            'GOOGLE_CLOUD_PROJECT': 'test-project'
        }):
            with patch('src.api.adk_integration.vertexai.init'):
                client = ParallaxPalADK()
                
                # Should use comprehensive mode as default
                query = client._prepare_query("Test", "invalid_mode")
                assert "thorough research" in query
    
    @pytest.mark.asyncio
    async def test_streaming_timeout(self):
        """Test handling of streaming timeouts"""
        
        with patch.dict('os.environ', {
            'GOOGLE_CLOUD_PROJECT': 'test-project'
        }):
            with patch('src.api.adk_integration.vertexai.init'):
                client = ParallaxPalADK()
                
                # Create session that hangs
                hanging_session = AsyncMock()
                
                async def hang_forever(query):
                    await asyncio.sleep(100)  # Simulate hang
                    yield Mock()  # Never reached
                
                hanging_session.stream_query = hang_forever
                
                with patch('src.api.adk_integration.StreamingSession', return_value=hanging_session):
                    # Use timeout to prevent actual hang in tests
                    with pytest.raises(asyncio.TimeoutError):
                        async with asyncio.timeout(1):
                            async for event in client.stream_research(
                                "Test", "user123", "session456"
                            ):
                                pass  # Should timeout before getting here


class TestADKPerformance:
    """Performance and load tests"""
    
    @pytest.mark.asyncio
    async def test_concurrent_research_streams(self):
        """Test handling multiple concurrent research streams"""
        
        with patch.dict('os.environ', {
            'GOOGLE_CLOUD_PROJECT': 'test-project'
        }):
            with patch('src.api.adk_integration.vertexai.init'):
                client = ParallaxPalADK()
                
                # Mock lightweight streaming
                mock_session = AsyncMock()
                
                async def mock_quick_stream(query):
                    yield Mock(type='start', agent_name='orchestrator', 
                              content='Start', progress=5, metadata={})
                    yield Mock(type='complete', agent_name='orchestrator',
                              content={'summary': 'Done'}, progress=100, metadata={})
                
                mock_session.stream_query = mock_quick_stream
                
                with patch('src.api.adk_integration.StreamingSession', return_value=mock_session):
                    # Run multiple concurrent streams
                    tasks = []
                    for i in range(10):
                        task = asyncio.create_task(
                            self._collect_events(
                                client,
                                f"Query {i}",
                                f"user{i}",
                                f"session{i}"
                            )
                        )
                        tasks.append(task)
                    
                    results = await asyncio.gather(*tasks)
                    
                    # All should complete successfully
                    assert len(results) == 10
                    for events in results:
                        assert len(events) == 2
                        assert events[-1]['type'] == 'complete'
    
    async def _collect_events(self, client, query, user_id, session_id):
        """Helper to collect all events from a stream"""
        events = []
        async for event in client.stream_research(query, user_id, session_id):
            events.append(event)
        return events
    
    @pytest.mark.asyncio
    async def test_memory_efficiency(self):
        """Test memory efficiency with large responses"""
        
        with patch.dict('os.environ', {
            'GOOGLE_CLOUD_PROJECT': 'test-project'
        }):
            with patch('src.api.adk_integration.vertexai.init'):
                client = ParallaxPalADK()
                
                # Create session with large response
                large_session = AsyncMock()
                
                async def mock_large_stream(query):
                    # Generate large knowledge graph
                    large_graph = {
                        'nodes': [
                            {'id': str(i), 'label': f'Node {i}', 'type': 'concept'}
                            for i in range(1000)
                        ],
                        'edges': [
                            {'source': str(i), 'target': str(i+1), 'type': 'relates'}
                            for i in range(999)
                        ]
                    }
                    
                    yield Mock(
                        type='complete',
                        agent_name='orchestrator',
                        content={
                            'summary': 'Large result',
                            'knowledge_graph': large_graph
                        },
                        progress=100,
                        metadata={}
                    )
                
                large_session.stream_query = mock_large_stream
                
                with patch('src.api.adk_integration.StreamingSession', return_value=large_session):
                    # Should handle large response without issues
                    event_count = 0
                    async for event in client.stream_research("Test", "user123", "session456"):
                        event_count += 1
                        # Verify large graph is present
                        if event['type'] == 'complete':
                            kg = event['content']['knowledge_graph']
                            assert len(kg['nodes']) == 1000
                            assert len(kg['edges']) == 999
                    
                    assert event_count == 1