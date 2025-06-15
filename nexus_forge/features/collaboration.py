"""
Collaborative research feature for Parallax Pal

Enables multiple users to collaborate on research projects with real-time
updates, shared knowledge graphs, and coordinated agent work.
"""

from typing import List, Dict, Set, Optional, Any
import asyncio
import uuid
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, asdict
from enum import Enum

from google.cloud import firestore
from ..state.distributed_state import DistributedStateManager
from ..security.validation import validate_user_id

logger = logging.getLogger(__name__)


class CollaborationRole(Enum):
    """Roles in a collaborative research session"""
    OWNER = "owner"
    EDITOR = "editor"
    VIEWER = "viewer"


class CollaborationPermission(Enum):
    """Permissions for collaboration actions"""
    CREATE_RESEARCH = "create_research"
    EDIT_RESEARCH = "edit_research"
    VIEW_RESEARCH = "view_research"
    INVITE_USERS = "invite_users"
    EXPORT_RESULTS = "export_results"
    DELETE_COLLABORATION = "delete_collaboration"


# Role-based permissions
ROLE_PERMISSIONS = {
    CollaborationRole.OWNER: [
        CollaborationPermission.CREATE_RESEARCH,
        CollaborationPermission.EDIT_RESEARCH,
        CollaborationPermission.VIEW_RESEARCH,
        CollaborationPermission.INVITE_USERS,
        CollaborationPermission.EXPORT_RESULTS,
        CollaborationPermission.DELETE_COLLABORATION
    ],
    CollaborationRole.EDITOR: [
        CollaborationPermission.CREATE_RESEARCH,
        CollaborationPermission.EDIT_RESEARCH,
        CollaborationPermission.VIEW_RESEARCH,
        CollaborationPermission.EXPORT_RESULTS
    ],
    CollaborationRole.VIEWER: [
        CollaborationPermission.VIEW_RESEARCH,
        CollaborationPermission.EXPORT_RESULTS
    ]
}


@dataclass
class CollaborationMember:
    """Member of a collaboration"""
    user_id: str
    role: CollaborationRole
    joined_at: str
    last_active: str
    contributions: int = 0


@dataclass
class CollaborationSession:
    """Collaborative research session"""
    id: str
    title: str
    description: str
    owner_id: str
    created_at: str
    updated_at: str
    status: str
    members: List[CollaborationMember]
    research_tasks: List[str]
    shared_graphs: List[Dict[str, Any]]
    settings: Dict[str, Any]


class CollaborativeResearchManager:
    """Manage collaborative research sessions"""
    
    def __init__(
        self,
        state_manager: DistributedStateManager,
        websocket_manager: Any,
        adk_integration: Any
    ):
        """
        Initialize collaborative research manager
        
        Args:
            state_manager: Distributed state manager
            websocket_manager: WebSocket manager for real-time updates
            adk_integration: ADK integration for agent coordination
        """
        self.state = state_manager
        self.websocket = websocket_manager
        self.adk = adk_integration
        
        # Active collaborations cache
        self.active_collaborations: Dict[str, CollaborationSession] = {}
        
        # User presence tracking
        self.user_presence: Dict[str, Set[str]] = {}  # collab_id -> set of active user_ids
        
        logger.info("Collaborative research manager initialized")
    
    async def create_collaboration(
        self,
        owner_id: str,
        title: str,
        description: str,
        settings: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a new collaborative research session
        
        Args:
            owner_id: User ID of the owner
            title: Collaboration title
            description: Collaboration description
            settings: Optional collaboration settings
            
        Returns:
            Collaboration ID
        """
        if not validate_user_id(owner_id):
            raise ValueError("Invalid user ID")
        
        collab_id = f"collab_{uuid.uuid4().hex[:12]}"
        now = datetime.now().isoformat()
        
        # Create owner member
        owner_member = CollaborationMember(
            user_id=owner_id,
            role=CollaborationRole.OWNER,
            joined_at=now,
            last_active=now
        )
        
        # Default settings
        default_settings = {
            'max_members': 10,
            'allow_anonymous': False,
            'require_approval': False,
            'auto_save': True,
            'share_mode': 'private',  # private, link, public
            'retention_days': 30
        }
        
        if settings:
            default_settings.update(settings)
        
        # Create collaboration session
        collaboration = CollaborationSession(
            id=collab_id,
            title=title,
            description=description,
            owner_id=owner_id,
            created_at=now,
            updated_at=now,
            status='active',
            members=[owner_member],
            research_tasks=[],
            shared_graphs=[],
            settings=default_settings
        )
        
        # Store in Firestore
        await self.state.firestore.collection('collaborations').document(
            collab_id
        ).set(asdict(collaboration))
        
        # Cache locally
        self.active_collaborations[collab_id] = collaboration
        
        # Initialize presence
        self.user_presence[collab_id] = {owner_id}
        
        # Publish creation event
        await self.state.publish_event(f"collab:{collab_id}", {
            'type': 'collaboration_created',
            'collaboration_id': collab_id,
            'owner_id': owner_id,
            'title': title
        })
        
        # Log metrics
        await self.state.increment_metric('collaborations_created')
        
        logger.info(f"Collaboration created: {collab_id} by {owner_id}")
        
        return collab_id
    
    async def join_collaboration(
        self,
        collab_id: str,
        user_id: str,
        invite_code: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Join an existing collaboration
        
        Args:
            collab_id: Collaboration ID
            user_id: User ID joining
            invite_code: Optional invite code
            
        Returns:
            Join result with collaboration data
        """
        if not validate_user_id(user_id):
            raise ValueError("Invalid user ID")
        
        # Get collaboration
        collaboration = await self._get_collaboration(collab_id)
        if not collaboration:
            return {
                'success': False,
                'error': 'Collaboration not found'
            }
        
        # Check if already a member
        if any(m.user_id == user_id for m in collaboration.members):
            return {
                'success': False,
                'error': 'Already a member'
            }
        
        # Check member limit
        if len(collaboration.members) >= collaboration.settings.get('max_members', 10):
            return {
                'success': False,
                'error': 'Collaboration is full'
            }
        
        # Validate invite if required
        if collaboration.settings.get('share_mode') == 'private':
            if not invite_code or not await self._validate_invite_code(collab_id, invite_code):
                return {
                    'success': False,
                    'error': 'Valid invite code required'
                }
        
        # Determine role
        role = CollaborationRole.VIEWER
        if collaboration.settings.get('new_member_role') == 'editor':
            role = CollaborationRole.EDITOR
        
        # Add member
        now = datetime.now().isoformat()
        new_member = CollaborationMember(
            user_id=user_id,
            role=role,
            joined_at=now,
            last_active=now
        )
        
        # Update Firestore
        await self.state.firestore.collection('collaborations').document(
            collab_id
        ).update({
            'members': firestore.ArrayUnion([asdict(new_member)]),
            'updated_at': now
        })
        
        # Update cache
        if collab_id in self.active_collaborations:
            self.active_collaborations[collab_id].members.append(new_member)
        
        # Add to presence
        if collab_id not in self.user_presence:
            self.user_presence[collab_id] = set()
        self.user_presence[collab_id].add(user_id)
        
        # Notify all members
        await self._broadcast_to_collaboration(collab_id, {
            'type': 'member_joined',
            'user_id': user_id,
            'role': role.value,
            'timestamp': now
        })
        
        # Return success with collaboration data
        return {
            'success': True,
            'collaboration': asdict(collaboration),
            'role': role.value
        }
    
    async def share_research(
        self,
        collab_id: str,
        user_id: str,
        task_id: str,
        include_graph: bool = True
    ) -> bool:
        """
        Share research results with collaboration
        
        Args:
            collab_id: Collaboration ID
            user_id: User sharing the research
            task_id: Research task ID
            include_graph: Whether to include knowledge graph
            
        Returns:
            Success status
        """
        # Check permissions
        if not await self._check_permission(
            collab_id,
            user_id,
            CollaborationPermission.CREATE_RESEARCH
        ):
            return False
        
        # Get research task data
        task_data = await self.state.get_session_state(f"task:{task_id}")
        if not task_data:
            return False
        
        # Prepare shared research entry
        shared_entry = {
            'id': str(uuid.uuid4()),
            'task_id': task_id,
            'shared_by': user_id,
            'shared_at': datetime.now().isoformat(),
            'query': task_data.get('query', ''),
            'summary': task_data.get('results', {}).get('summary', ''),
            'sources': task_data.get('results', {}).get('sources', []),
            'insights': task_data.get('results', {}).get('insights', [])
        }
        
        # Include knowledge graph if requested
        if include_graph and 'knowledge_graph' in task_data.get('results', {}):
            shared_entry['knowledge_graph'] = task_data['results']['knowledge_graph']
            
            # Add to shared graphs
            await self.state.firestore.collection('collaborations').document(
                collab_id
            ).update({
                'shared_graphs': firestore.ArrayUnion([{
                    'id': shared_entry['id'],
                    'title': task_data.get('query', 'Untitled'),
                    'graph_data': shared_entry['knowledge_graph'],
                    'created_at': shared_entry['shared_at']
                }])
            })
        
        # Add to research tasks
        await self.state.firestore.collection('collaborations').document(
            collab_id
        ).update({
            'research_tasks': firestore.ArrayUnion([shared_entry['id']]),
            'updated_at': datetime.now().isoformat()
        })
        
        # Store full research data
        await self.state.firestore.collection('shared_research').document(
            shared_entry['id']
        ).set(shared_entry)
        
        # Update member contribution count
        await self._increment_contribution(collab_id, user_id)
        
        # Broadcast to all members
        await self._broadcast_to_collaboration(collab_id, {
            'type': 'research_shared',
            'user_id': user_id,
            'research_id': shared_entry['id'],
            'query': shared_entry['query'],
            'has_graph': include_graph
        })
        
        logger.info(f"Research {task_id} shared to collaboration {collab_id}")
        
        return True
    
    async def coordinate_research(
        self,
        collab_id: str,
        query: str,
        assigned_users: Optional[List[str]] = None,
        subtasks: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Coordinate multi-user research with task distribution
        
        Args:
            collab_id: Collaboration ID
            query: Main research query
            assigned_users: Optional list of users to assign subtasks
            subtasks: Optional predefined subtasks
            
        Returns:
            Coordination task ID
        """
        collaboration = await self._get_collaboration(collab_id)
        if not collaboration:
            raise ValueError("Collaboration not found")
        
        coordination_id = f"coord_{uuid.uuid4().hex[:12]}"
        
        # If no subtasks provided, use ADK to decompose query
        if not subtasks:
            decomposition_result = await self.adk.agents['orchestrator'].aquery(
                f"Decompose this research query into 3-5 focused subtasks: {query}"
            )
            # Parse subtasks from result
            subtasks = self._parse_subtasks(decomposition_result)
        
        # Assign users to subtasks
        active_members = [
            m for m in collaboration.members
            if m.role in [CollaborationRole.OWNER, CollaborationRole.EDITOR]
        ]
        
        if not assigned_users:
            assigned_users = [m.user_id for m in active_members]
        
        # Create coordination task
        coordination_task = {
            'id': coordination_id,
            'collaboration_id': collab_id,
            'main_query': query,
            'created_at': datetime.now().isoformat(),
            'status': 'active',
            'subtasks': []
        }
        
        # Distribute subtasks
        for i, subtask in enumerate(subtasks):
            assigned_user = assigned_users[i % len(assigned_users)] if assigned_users else None
            
            subtask_entry = {
                'id': f"{coordination_id}_sub_{i}",
                'query': subtask.get('query', ''),
                'assigned_to': assigned_user,
                'status': 'pending',
                'created_at': datetime.now().isoformat()
            }
            
            coordination_task['subtasks'].append(subtask_entry)
            
            # Notify assigned user
            if assigned_user:
                await self._notify_user(assigned_user, {
                    'type': 'task_assigned',
                    'coordination_id': coordination_id,
                    'subtask_id': subtask_entry['id'],
                    'query': subtask_entry['query']
                })
        
        # Store coordination task
        await self.state.firestore.collection('coordination_tasks').document(
            coordination_id
        ).set(coordination_task)
        
        # Broadcast to collaboration
        await self._broadcast_to_collaboration(collab_id, {
            'type': 'coordination_started',
            'coordination_id': coordination_id,
            'main_query': query,
            'subtask_count': len(subtasks)
        })
        
        return coordination_id
    
    async def merge_knowledge_graphs(
        self,
        collab_id: str,
        graph_ids: List[str]
    ) -> Dict[str, Any]:
        """
        Merge multiple knowledge graphs from collaboration
        
        Args:
            collab_id: Collaboration ID
            graph_ids: List of graph IDs to merge
            
        Returns:
            Merged knowledge graph
        """
        collaboration = await self._get_collaboration(collab_id)
        if not collaboration:
            return {'error': 'Collaboration not found'}
        
        # Collect graphs
        graphs = []
        for graph_id in graph_ids:
            graph_data = next(
                (g for g in collaboration.shared_graphs if g['id'] == graph_id),
                None
            )
            if graph_data:
                graphs.append(graph_data['graph_data'])
        
        if not graphs:
            return {'error': 'No graphs found'}
        
        # Merge nodes and edges
        merged_nodes = {}
        merged_edges = []
        
        for graph in graphs:
            # Merge nodes (deduplicate by label)
            for node in graph.get('nodes', []):
                node_key = f"{node['type']}:{node['label']}"
                if node_key not in merged_nodes:
                    merged_nodes[node_key] = node
                else:
                    # Merge properties
                    existing = merged_nodes[node_key]
                    for key, value in node.get('properties', {}).items():
                        if key not in existing.get('properties', {}):
                            existing['properties'][key] = value
            
            # Collect edges
            merged_edges.extend(graph.get('edges', []))
        
        # Deduplicate edges
        unique_edges = []
        edge_keys = set()
        
        for edge in merged_edges:
            edge_key = f"{edge['source']}:{edge['target']}:{edge['type']}"
            if edge_key not in edge_keys:
                edge_keys.add(edge_key)
                unique_edges.append(edge)
            else:
                # Update weight for duplicate edges
                for e in unique_edges:
                    if (e['source'] == edge['source'] and 
                        e['target'] == edge['target'] and 
                        e['type'] == edge['type']):
                        e['weight'] = max(e.get('weight', 0), edge.get('weight', 0))
        
        # Create merged graph
        merged_graph = {
            'nodes': list(merged_nodes.values()),
            'edges': unique_edges,
            'metadata': {
                'merged_from': graph_ids,
                'merge_date': datetime.now().isoformat(),
                'node_count': len(merged_nodes),
                'edge_count': len(unique_edges)
            }
        }
        
        # Store merged graph
        merged_id = f"merged_{uuid.uuid4().hex[:8]}"
        await self.state.firestore.collection('collaborations').document(
            collab_id
        ).update({
            'shared_graphs': firestore.ArrayUnion([{
                'id': merged_id,
                'title': f"Merged Graph ({len(graph_ids)} sources)",
                'graph_data': merged_graph,
                'created_at': datetime.now().isoformat()
            }])
        })
        
        return merged_graph
    
    async def get_collaboration_analytics(
        self,
        collab_id: str
    ) -> Dict[str, Any]:
        """
        Get analytics for a collaboration
        
        Args:
            collab_id: Collaboration ID
            
        Returns:
            Analytics data
        """
        collaboration = await self._get_collaboration(collab_id)
        if not collaboration:
            return {'error': 'Collaboration not found'}
        
        # Calculate member statistics
        member_stats = []
        for member in collaboration.members:
            stats = {
                'user_id': member.user_id,
                'role': member.role.value,
                'joined_at': member.joined_at,
                'contributions': member.contributions,
                'days_active': (
                    datetime.now() - datetime.fromisoformat(member.joined_at)
                ).days
            }
            member_stats.append(stats)
        
        # Get research statistics
        research_count = len(collaboration.research_tasks)
        graph_count = len(collaboration.shared_graphs)
        
        # Calculate activity timeline
        activity_timeline = await self._get_activity_timeline(collab_id)
        
        analytics = {
            'collaboration_id': collab_id,
            'created_at': collaboration.created_at,
            'member_count': len(collaboration.members),
            'member_stats': member_stats,
            'research_count': research_count,
            'graph_count': graph_count,
            'total_contributions': sum(m.contributions for m in collaboration.members),
            'activity_timeline': activity_timeline,
            'most_active_member': max(
                member_stats,
                key=lambda x: x['contributions']
            ) if member_stats else None,
            'collaboration_age_days': (
                datetime.now() - datetime.fromisoformat(collaboration.created_at)
            ).days
        }
        
        return analytics
    
    # Helper methods
    
    async def _get_collaboration(self, collab_id: str) -> Optional[CollaborationSession]:
        """Get collaboration from cache or Firestore"""
        
        # Check cache
        if collab_id in self.active_collaborations:
            return self.active_collaborations[collab_id]
        
        # Load from Firestore
        doc = await self.state.firestore.collection('collaborations').document(
            collab_id
        ).get()
        
        if doc.exists:
            data = doc.to_dict()
            # Convert members to CollaborationMember objects
            data['members'] = [
                CollaborationMember(**{**m, 'role': CollaborationRole(m['role'])})
                for m in data['members']
            ]
            collaboration = CollaborationSession(**data)
            self.active_collaborations[collab_id] = collaboration
            return collaboration
        
        return None
    
    async def _check_permission(
        self,
        collab_id: str,
        user_id: str,
        permission: CollaborationPermission
    ) -> bool:
        """Check if user has permission in collaboration"""
        
        collaboration = await self._get_collaboration(collab_id)
        if not collaboration:
            return False
        
        member = next((m for m in collaboration.members if m.user_id == user_id), None)
        if not member:
            return False
        
        allowed_permissions = ROLE_PERMISSIONS.get(member.role, [])
        return permission in allowed_permissions
    
    async def _broadcast_to_collaboration(
        self,
        collab_id: str,
        message: Dict[str, Any]
    ):
        """Broadcast message to all collaboration members"""
        
        collaboration = await self._get_collaboration(collab_id)
        if not collaboration:
            return
        
        # Add collaboration context
        message['collaboration_id'] = collab_id
        message['timestamp'] = datetime.now().isoformat()
        
        # Send to all members
        for member in collaboration.members:
            if member.user_id in self.user_presence.get(collab_id, set()):
                await self.websocket.broadcast_to_user(member.user_id, message)
    
    async def _notify_user(self, user_id: str, notification: Dict[str, Any]):
        """Send notification to specific user"""
        
        notification['timestamp'] = datetime.now().isoformat()
        await self.websocket.broadcast_to_user(user_id, notification)
    
    async def _increment_contribution(self, collab_id: str, user_id: str):
        """Increment user's contribution count"""
        
        # Update in Firestore
        doc_ref = self.state.firestore.collection('collaborations').document(collab_id)
        
        # Use transaction to safely increment
        @firestore.transactional
        async def update_contribution(transaction, doc_ref):
            doc = await doc_ref.get(transaction=transaction)
            if doc.exists:
                members = doc.to_dict()['members']
                for i, member in enumerate(members):
                    if member['user_id'] == user_id:
                        members[i]['contributions'] = member.get('contributions', 0) + 1
                        members[i]['last_active'] = datetime.now().isoformat()
                        break
                
                transaction.update(doc_ref, {'members': members})
        
        transaction = self.state.firestore.transaction()
        await update_contribution(transaction, doc_ref)
        
        # Update cache
        if collab_id in self.active_collaborations:
            for member in self.active_collaborations[collab_id].members:
                if member.user_id == user_id:
                    member.contributions += 1
                    member.last_active = datetime.now().isoformat()
                    break
    
    def _parse_subtasks(self, decomposition_result: str) -> List[Dict[str, str]]:
        """Parse subtasks from ADK decomposition result"""
        
        # Simple parsing - in production, use more sophisticated NLP
        lines = decomposition_result.strip().split('\n')
        subtasks = []
        
        for line in lines:
            if line.strip() and (line[0].isdigit() or line.startswith('-')):
                # Remove numbering and clean
                query = line.strip().lstrip('0123456789.-) ').strip()
                if query:
                    subtasks.append({'query': query})
        
        return subtasks[:5]  # Limit to 5 subtasks
    
    async def _validate_invite_code(self, collab_id: str, invite_code: str) -> bool:
        """Validate invite code for collaboration"""
        
        # Check if invite code exists and is valid
        doc = await self.state.firestore.collection('invite_codes').document(
            invite_code
        ).get()
        
        if not doc.exists:
            return False
        
        invite_data = doc.to_dict()
        
        # Check expiration
        if 'expires_at' in invite_data:
            if datetime.fromisoformat(invite_data['expires_at']) < datetime.now():
                return False
        
        # Check collaboration match
        return invite_data.get('collaboration_id') == collab_id
    
    async def _get_activity_timeline(
        self,
        collab_id: str,
        days: int = 7
    ) -> List[Dict[str, Any]]:
        """Get activity timeline for collaboration"""
        
        # Query activity logs
        start_date = datetime.now() - timedelta(days=days)
        
        activities = []
        
        # Get recent shared research
        query = self.state.firestore.collection('shared_research').where(
            'collaboration_id', '==', collab_id
        ).where(
            'shared_at', '>=', start_date.isoformat()
        ).order_by('shared_at', direction=firestore.Query.DESCENDING).limit(50)
        
        docs = await query.get()
        
        for doc in docs:
            data = doc.to_dict()
            activities.append({
                'type': 'research_shared',
                'timestamp': data['shared_at'],
                'user_id': data['shared_by'],
                'details': {
                    'query': data['query']
                }
            })
        
        # Sort by timestamp
        activities.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return activities