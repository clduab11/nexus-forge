"""
Workflow Builder API endpoints
"""

import json
from typing import Any, Dict, List, Optional

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import StreamingResponse

from ...core.auth import get_current_user
from ...core.exceptions import NotFoundError, ValidationError
from ...websockets.manager import WebSocketManager
from ...workflow_builder import (
    ExecutionStatus,
    Workflow,
    WorkflowEngine,
    WorkflowExecution,
    WorkflowTemplate,
)
from ..schemas.auth import User

router = APIRouter(prefix="/api/v1/workflows", tags=["workflows"])

# Initialize services
workflow_engine = WorkflowEngine()
websocket_manager = WebSocketManager()


# Workflow Management


@router.post("/", response_model=Workflow)
async def create_workflow(
    name: str,
    description: str,
    organization: Optional[str] = None,
    current_user: User = Depends(get_current_user),
):
    """Create a new workflow"""
    try:
        workflow = await workflow_engine.create_workflow(
            name=name,
            description=description,
            author_id=current_user.id,
            organization=organization,
        )
        return workflow
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{workflow_id}", response_model=Workflow)
async def get_workflow(workflow_id: str):
    """Get workflow by ID"""
    try:
        workflow = await workflow_engine.get_workflow(workflow_id)
        return workflow
    except NotFoundError:
        raise HTTPException(status_code=404, detail="Workflow not found")


@router.put("/{workflow_id}", response_model=Workflow)
async def update_workflow(
    workflow_id: str,
    workflow_data: Dict[str, Any],
    current_user: User = Depends(get_current_user),
):
    """Update an existing workflow"""
    try:
        # Get existing workflow
        workflow = await workflow_engine.get_workflow(workflow_id)

        # Check ownership
        if workflow.author_id != current_user.id:
            raise HTTPException(status_code=403, detail="Access denied")

        # Update workflow data
        for key, value in workflow_data.items():
            if hasattr(workflow, key):
                setattr(workflow, key, value)

        # Save updates
        updated_workflow = await workflow_engine.update_workflow(workflow)
        return updated_workflow

    except NotFoundError:
        raise HTTPException(status_code=404, detail="Workflow not found")
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/{workflow_id}")
async def delete_workflow(
    workflow_id: str, current_user: User = Depends(get_current_user)
):
    """Delete a workflow"""
    try:
        # Get workflow to check ownership
        workflow = await workflow_engine.get_workflow(workflow_id)

        if workflow.author_id != current_user.id:
            raise HTTPException(status_code=403, detail="Access denied")

        success = await workflow_engine.delete_workflow(workflow_id)

        if success:
            return {"message": "Workflow deleted successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to delete workflow")

    except NotFoundError:
        raise HTTPException(status_code=404, detail="Workflow not found")


@router.get("/", response_model=List[Workflow])
async def list_workflows(
    author_id: Optional[str] = Query(None),
    organization: Optional[str] = Query(None),
    category: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_current_user),
):
    """List workflows with filtering"""
    # If no author_id specified, show user's own workflows
    if not author_id:
        author_id = current_user.id

    workflows = await workflow_engine.list_workflows(
        author_id=author_id,
        organization=organization,
        category=category,
        limit=limit,
        offset=offset,
    )

    return workflows


# Node Type Management


@router.get("/node-types", response_model=Dict[str, Dict[str, Any]])
async def get_node_types():
    """Get all available node types"""
    return workflow_engine.get_node_types()


@router.get("/node-types/{node_type}")
async def get_node_type(node_type: str):
    """Get specific node type definition"""
    try:
        return workflow_engine.get_node_type(node_type)
    except NotFoundError:
        raise HTTPException(status_code=404, detail="Node type not found")


# Workflow Validation and Compilation


@router.post("/{workflow_id}/validate")
async def validate_workflow(workflow_id: str):
    """Validate workflow structure and configuration"""
    try:
        workflow = await workflow_engine.get_workflow(workflow_id)
        validation_result = await workflow_engine.validate_workflow(workflow)
        return validation_result
    except NotFoundError:
        raise HTTPException(status_code=404, detail="Workflow not found")


@router.post("/{workflow_id}/compile")
async def compile_workflow(workflow_id: str):
    """Compile workflow for execution"""
    try:
        workflow = await workflow_engine.get_workflow(workflow_id)
        compiled = await workflow_engine.compile_workflow(workflow)

        return {
            "workflow_id": compiled.workflow_id,
            "entry_points": compiled.entry_points,
            "total_nodes": len(compiled.node_mapping),
            "execution_order": compiled.execution_order[:10],  # First 10 nodes
            "variables": list(compiled.variables.keys()),
        }
    except NotFoundError:
        raise HTTPException(status_code=404, detail="Workflow not found")
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))


# Workflow Execution


@router.post("/{workflow_id}/execute", response_model=WorkflowExecution)
async def execute_workflow(
    workflow_id: str,
    trigger_data: Dict[str, Any],
    current_user: User = Depends(get_current_user),
):
    """Execute a workflow"""
    try:
        execution = await workflow_engine.execute_workflow(
            workflow_id=workflow_id, trigger_data=trigger_data, user_id=current_user.id
        )
        return execution
    except NotFoundError:
        raise HTTPException(status_code=404, detail="Workflow not found")
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{workflow_id}/executions", response_model=List[WorkflowExecution])
async def list_executions(
    workflow_id: str,
    status: Optional[ExecutionStatus] = Query(None),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_current_user),
):
    """List workflow executions"""
    executions = await workflow_engine.list_executions(
        workflow_id=workflow_id,
        user_id=current_user.id,
        status=status,
        limit=limit,
        offset=offset,
    )

    return executions


@router.get("/executions/{execution_id}", response_model=WorkflowExecution)
async def get_execution(execution_id: str):
    """Get workflow execution by ID"""
    execution = await workflow_engine.get_execution(execution_id)

    if not execution:
        raise HTTPException(status_code=404, detail="Execution not found")

    return execution


# Execution Control


@router.post("/executions/{execution_id}/pause")
async def pause_execution(execution_id: str):
    """Pause workflow execution"""
    success = await workflow_engine.pause_execution(execution_id)

    if success:
        return {"message": "Execution paused"}
    else:
        raise HTTPException(
            status_code=404, detail="Execution not found or cannot be paused"
        )


@router.post("/executions/{execution_id}/resume")
async def resume_execution(execution_id: str):
    """Resume paused workflow execution"""
    success = await workflow_engine.resume_execution(execution_id)

    if success:
        return {"message": "Execution resumed"}
    else:
        raise HTTPException(
            status_code=404, detail="Execution not found or cannot be resumed"
        )


@router.post("/executions/{execution_id}/cancel")
async def cancel_execution(execution_id: str):
    """Cancel workflow execution"""
    success = await workflow_engine.cancel_execution(execution_id)

    if success:
        return {"message": "Execution cancelled"}
    else:
        raise HTTPException(
            status_code=404, detail="Execution not found or cannot be cancelled"
        )


# Real-time Execution Monitoring


@router.websocket("/executions/{execution_id}/monitor")
async def monitor_execution(websocket: WebSocket, execution_id: str):
    """Monitor workflow execution in real-time"""
    await websocket.accept()

    try:
        # Add client to websocket manager
        client_id = f"exec_monitor_{execution_id}_{id(websocket)}"
        await websocket_manager.connect(client_id, websocket)

        # Send initial execution state
        execution = await workflow_engine.get_execution(execution_id)
        if execution:
            await websocket.send_json(
                {"type": "execution_state", "execution": execution.dict()}
            )

        # Keep connection alive and listen for messages
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)

                # Handle client commands
                if message.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})

            except WebSocketDisconnect:
                break
            except Exception as e:
                await websocket.send_json({"type": "error", "message": str(e)})

    except WebSocketDisconnect:
        pass
    finally:
        await websocket_manager.disconnect(client_id)


# Template Management


@router.get("/templates", response_model=List[WorkflowTemplate])
async def list_templates(
    category: Optional[str] = Query(None),
    difficulty: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=100),
):
    """List workflow templates"""
    templates = await workflow_engine.list_templates(
        category=category, difficulty=difficulty, limit=limit
    )

    return templates


@router.get("/templates/{template_id}", response_model=WorkflowTemplate)
async def get_template(template_id: str):
    """Get workflow template by ID"""
    try:
        template = await workflow_engine.get_template(template_id)
        return template
    except NotFoundError:
        raise HTTPException(status_code=404, detail="Template not found")


@router.post("/templates/{template_id}/create-workflow", response_model=Workflow)
async def create_from_template(
    template_id: str,
    name: str,
    config: Dict[str, Any],
    current_user: User = Depends(get_current_user),
):
    """Create workflow from template"""
    try:
        workflow = await workflow_engine.create_from_template(
            template_id=template_id, name=name, author_id=current_user.id, config=config
        )
        return workflow
    except NotFoundError:
        raise HTTPException(status_code=404, detail="Template not found")


@router.post("/templates", response_model=WorkflowTemplate)
async def create_template(
    name: str,
    description: str,
    category: str,
    workflow_id: str,
    config_schema: Optional[Dict[str, Any]] = None,
    current_user: User = Depends(get_current_user),
):
    """Create workflow template"""
    try:
        # Get workflow
        workflow = await workflow_engine.get_workflow(workflow_id)

        # Check ownership
        if workflow.author_id != current_user.id:
            raise HTTPException(status_code=403, detail="Access denied")

        template = await workflow_engine.create_template(
            name=name,
            description=description,
            category=category,
            workflow=workflow,
            author=current_user.email,
            config_schema=config_schema,
        )

        return template

    except NotFoundError:
        raise HTTPException(status_code=404, detail="Workflow not found")


# Statistics and Analytics


@router.get("/{workflow_id}/stats")
async def get_workflow_stats(workflow_id: str):
    """Get workflow statistics"""
    try:
        stats = await workflow_engine.get_workflow_stats(workflow_id)
        return stats
    except NotFoundError:
        raise HTTPException(status_code=404, detail="Workflow not found")


# Export and Import


@router.get("/{workflow_id}/export")
async def export_workflow(workflow_id: str):
    """Export workflow as JSON"""
    try:
        workflow = await workflow_engine.get_workflow(workflow_id)

        # Create export data
        export_data = {
            "format_version": "1.0",
            "workflow": workflow.dict(),
            "node_types": {
                node_type: workflow_engine.get_node_type(node_type)
                for node in workflow.nodes
                for node_type in [node.type.value]
            },
            "exported_at": workflow.updated_at.isoformat(),
        }

        # Return as downloadable JSON
        def generate():
            yield json.dumps(export_data, indent=2)

        return StreamingResponse(
            generate(),
            media_type="application/json",
            headers={
                "Content-Disposition": f"attachment; filename={workflow.name}.json"
            },
        )

    except NotFoundError:
        raise HTTPException(status_code=404, detail="Workflow not found")


@router.post("/import")
async def import_workflow(
    workflow_data: Dict[str, Any], current_user: User = Depends(get_current_user)
):
    """Import workflow from JSON"""
    try:
        # Validate import format
        if "workflow" not in workflow_data:
            raise HTTPException(status_code=400, detail="Invalid workflow format")

        # Extract workflow
        imported_workflow_data = workflow_data["workflow"]

        # Update metadata
        imported_workflow_data.update(
            {
                "id": None,  # Generate new ID
                "author_id": current_user.id,
                "created_at": None,
                "updated_at": None,
            }
        )

        # Create workflow
        workflow = Workflow(**imported_workflow_data)
        await workflow_engine._save_workflow(workflow)

        return workflow

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Import failed: {str(e)}")


# Workflow Testing


@router.post("/{workflow_id}/test")
async def test_workflow(
    workflow_id: str,
    test_data: Dict[str, Any],
    current_user: User = Depends(get_current_user),
):
    """Test workflow with sample data"""
    try:
        # Execute workflow in test mode
        execution = await workflow_engine.execute_workflow(
            workflow_id=workflow_id, trigger_data=test_data, user_id=current_user.id
        )

        # Return execution summary
        return {
            "execution_id": execution.id,
            "status": execution.status,
            "duration_ms": execution.duration_ms,
            "nodes_executed": execution.nodes_executed,
            "output": execution.output,
            "error": execution.error,
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Test failed: {str(e)}")


# Workflow Sharing


@router.post("/{workflow_id}/share")
async def share_workflow(
    workflow_id: str,
    public: bool = False,
    current_user: User = Depends(get_current_user),
):
    """Share workflow publicly or with organization"""
    try:
        workflow = await workflow_engine.get_workflow(workflow_id)

        # Check ownership
        if workflow.author_id != current_user.id:
            raise HTTPException(status_code=403, detail="Access denied")

        # Update sharing settings
        workflow.public = public
        await workflow_engine.update_workflow(workflow)

        return {
            "message": f"Workflow {'shared publicly' if public else 'made private'}",
            "public": public,
        }

    except NotFoundError:
        raise HTTPException(status_code=404, detail="Workflow not found")


@router.get("/public", response_model=List[Workflow])
async def list_public_workflows(
    category: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    """List public workflows"""
    # This would query for public workflows
    # For now, return empty list
    return []
