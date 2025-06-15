# Nexus Forge API Reference

## 🎯 Overview

The Nexus Forge API provides programmatic access to the one-shot app builder functionality. This RESTful API allows developers to integrate app generation capabilities into their own applications and workflows.

**Base URL**: `https://api.nexusforge.example.com`

**API Version**: `v1`

**Authentication**: Bearer token (JWT)

## 🔐 Authentication

### POST /auth/login
Authenticate user and receive access token.

**Request Body:**
```json
{
  "email": "user@example.com",
  "password": "secure_password"
}
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "def50200ace...",
  "token_type": "bearer",
  "expires_in": 1800
}
```

### POST /auth/refresh
Refresh expired access token.

**Request Body:**
```json
{
  "refresh_token": "def50200ace..."
}
```

### POST /auth/logout
Invalidate current session.

**Headers:**
```
Authorization: Bearer <access_token>
```

## 🏗️ App Building API

### POST /api/nexus-forge/build
Start a new app building session.

**Headers:**
```
Authorization: Bearer <access_token>
Content-Type: application/json
```

**Request Body:**
```json
{
  "prompt": "Build a real-time analytics dashboard with charts and user authentication",
  "config": {
    "useAdaptiveThinking": true,
    "enableVideoDemo": true,
    "deployToCloudRun": true,
    "techStack": {
      "frontend": "react",
      "backend": "fastapi",
      "database": "postgresql"
    },
    "styling": "tailwind",
    "authentication": "oauth2"
  }
}
```

**Response:**
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "build_started",
  "message": "App building session started. Connect to WebSocket for real-time updates.",
  "websocket_url": "/ws/nexus-forge/550e8400-e29b-41d4-a716-446655440000",
  "estimated_duration": "4-6 minutes"
}
```

**Error Responses:**
```json
// 400 Bad Request
{
  "detail": "App description is required",
  "error_code": "INVALID_INPUT"
}

// 429 Too Many Requests
{
  "detail": "Rate limit exceeded. Please wait before starting another build.",
  "retry_after": 60
}
```

### GET /api/nexus-forge/build/{session_id}
Get the status and results of a build session.

**Parameters:**
- `session_id` (string): The unique session identifier

**Response:**
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "started_at": "2024-01-15T10:00:00Z",
  "completed_at": "2024-01-15T10:04:23Z",
  "progress": 100,
  "current_phase": "deployment",
  "result": {
    "specification": {
      "name": "Analytics Dashboard Pro",
      "description": "Real-time analytics dashboard with interactive charts",
      "features": [
        "Real-time data visualization",
        "User authentication",
        "Export functionality",
        "Interactive charts"
      ],
      "tech_stack": {
        "frontend": "React",
        "backend": "FastAPI",
        "database": "PostgreSQL"
      },
      "ui_components": ["Dashboard", "Charts", "UserProfile", "Settings"],
      "api_endpoints": [
        {
          "method": "GET",
          "path": "/api/data",
          "description": "Fetch analytics data"
        },
        {
          "method": "POST",
          "path": "/api/auth",
          "description": "User authentication"
        }
      ]
    },
    "mockups": {
      "Dashboard": "https://storage.googleapis.com/nexus-forge-mockups/dashboard_modern.png",
      "Charts": "https://storage.googleapis.com/nexus-forge-mockups/charts_interactive.png",
      "UserProfile": "https://storage.googleapis.com/nexus-forge-mockups/userprofile_clean.png",
      "Settings": "https://storage.googleapis.com/nexus-forge-mockups/settings_organized.png"
    },
    "demo_video": "https://storage.googleapis.com/nexus-forge-demos/analytics_dashboard_pro_demo.mp4",
    "code_files": {
      "backend/main.py": "from fastapi import FastAPI...",
      "frontend/src/App.tsx": "import React from 'react'...",
      "tests/test_main.py": "import pytest...",
      "requirements.txt": "fastapi==0.104.1\nuvicorn...",
      "package.json": "{\"name\": \"analytics-dashboard\"...}"
    },
    "deployment_url": "https://analytics-dashboard-abc123.run.app",
    "build_time": "4 minutes 23 seconds",
    "orchestrator": "Starri",
    "models_used": ["gemini_pro", "imagen", "veo", "jules", "gemini_flash"]
  }
}
```

**Status Values:**
- `initializing`: Session created, waiting to start
- `analyzing`: Starri analyzing the prompt
- `generating_spec`: Gemini 2.5 Pro creating specification
- `building`: Parallel AI model execution
- `optimizing`: Gemini 2.5 Flash optimization
- `deploying`: Cloud Run deployment
- `completed`: Build finished successfully
- `failed`: Build encountered an error
- `cancelled`: User cancelled the build

### GET /api/nexus-forge/builds
List user's build sessions with pagination.

**Query Parameters:**
- `skip` (int, default: 0): Number of records to skip
- `limit` (int, default: 20): Maximum number of records to return
- `status` (string, optional): Filter by status
- `search` (string, optional): Search in app names and descriptions

**Response:**
```json
[
  {
    "session_id": "550e8400-e29b-41d4-a716-446655440000",
    "prompt": "Build a real-time analytics dashboard...",
    "status": "completed",
    "started_at": "2024-01-15T10:00:00Z",
    "completed_at": "2024-01-15T10:04:23Z",
    "app_name": "Analytics Dashboard Pro",
    "deployment_url": "https://analytics-dashboard-abc123.run.app"
  },
  {
    "session_id": "661e8400-e29b-41d4-a716-446655440111",
    "prompt": "Create an e-commerce platform...",
    "status": "building",
    "started_at": "2024-01-15T11:00:00Z",
    "progress": 65,
    "current_phase": "building"
  }
]
```

### POST /api/nexus-forge/deploy/{session_id}
Deploy a completed app to production.

**Request Body:**
```json
{
  "environment": "production",
  "domain": "my-app.example.com",
  "scaling": {
    "min_instances": 1,
    "max_instances": 10,
    "cpu": "1000m",
    "memory": "2Gi"
  }
}
```

**Response:**
```json
{
  "deployment_id": "deploy_123456789",
  "status": "deployed",
  "deployment_url": "https://my-app.example.com",
  "message": "App successfully deployed to production"
}
```

### DELETE /api/nexus-forge/build/{session_id}
Cancel a running build or delete a completed session.

**Response:**
```json
{
  "message": "Build session cancelled successfully",
  "status": "cancelled"
}
```

## 📋 Templates API

### GET /api/nexus-forge/templates
Get available app templates.

**Response:**
```json
[
  {
    "id": "analytics_dashboard",
    "name": "Analytics Dashboard",
    "description": "Real-time data visualization with charts and KPIs",
    "category": "Business Intelligence",
    "example_prompt": "Build an analytics dashboard that shows real-time sales data with interactive charts",
    "preview_image": "https://storage.googleapis.com/nexus-forge-templates/analytics_dashboard.png",
    "estimated_build_time": "3-5 minutes",
    "features": [
      "Real-time charts",
      "Data export",
      "User authentication",
      "Responsive design"
    ]
  },
  {
    "id": "ecommerce_store",
    "name": "E-Commerce Store",
    "description": "Complete online store with product catalog and payments",
    "category": "E-Commerce",
    "example_prompt": "Create an online store for selling handmade jewelry with product catalog and shopping cart",
    "preview_image": "https://storage.googleapis.com/nexus-forge-templates/ecommerce_store.png",
    "estimated_build_time": "5-7 minutes",
    "features": [
      "Product catalog",
      "Shopping cart",
      "Payment processing",
      "Order management"
    ]
  }
]
```

### POST /api/nexus-forge/build/from-template
Start a build using a template.

**Request Body:**
```json
{
  "template_id": "analytics_dashboard",
  "customizations": {
    "app_name": "Sales Analytics Pro",
    "color_scheme": "blue",
    "data_sources": ["postgresql", "api"],
    "features": ["export_pdf", "email_reports"]
  }
}
```

## 🔄 WebSocket API

### WebSocket Connection: /ws/nexus-forge/{session_id}

Connect to receive real-time build updates.

**Connection URL:**
```
wss://api.nexusforge.example.com/ws/nexus-forge/550e8400-e29b-41d4-a716-446655440000
```

**Message Types:**

#### Connection Confirmation
```json
{
  "type": "connected",
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "message": "Connected to Nexus Forge build session"
}
```

#### Progress Updates
```json
{
  "type": "progress_update",
  "phase": "generating_spec",
  "progress": 25,
  "message": "Starri analyzing app requirements...",
  "model": "starri",
  "timestamp": "2024-01-15T10:01:00Z"
}
```

#### Phase Completion
```json
{
  "type": "phase_completed",
  "phase": "building",
  "message": "AI models have completed parallel processing",
  "results": {
    "mockups_generated": 4,
    "video_created": true,
    "code_files": 12
  },
  "next_phase": "optimizing"
}
```

#### Build Completion
```json
{
  "type": "build_completed",
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "deployment_url": "https://analytics-dashboard-abc123.run.app",
  "total_time": "4 minutes 23 seconds",
  "summary": {
    "specification": "Generated",
    "mockups": 4,
    "demo_video": "Created",
    "code_files": 12,
    "deployment": "Successful"
  }
}
```

#### Error Notifications
```json
{
  "type": "error",
  "phase": "deployment",
  "error_code": "DEPLOYMENT_FAILED",
  "message": "Deployment failed due to resource constraints",
  "retry_possible": true
}
```

## 📊 Monitoring API

### GET /health
System health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2024-01-15T10:00:00Z",
  "environment": "production",
  "services": {
    "database": "healthy",
    "redis": "healthy",
    "ai_models": "healthy",
    "storage": "healthy"
  }
}
```

### GET /api/metrics
Get system metrics (requires admin role).

**Response:**
```json
{
  "builds": {
    "total": 1250,
    "successful": 1187,
    "failed": 63,
    "success_rate": 0.95
  },
  "performance": {
    "avg_build_time": "4.2 minutes",
    "avg_response_time": "150ms",
    "uptime": "99.8%"
  },
  "usage": {
    "active_sessions": 12,
    "daily_builds": 89,
    "monthly_builds": 2543
  }
}
```

## 🎛️ Configuration API

### GET /api/config/models
Get available AI models and their status.

**Response:**
```json
{
  "models": [
    {
      "name": "gemini-2.5-pro",
      "type": "specification",
      "status": "available",
      "version": "2.5",
      "capabilities": ["adaptive_thinking", "multi_tool_use"]
    },
    {
      "name": "imagen-4",
      "type": "image_generation",
      "status": "available",
      "version": "4.0",
      "max_resolution": "2K"
    },
    {
      "name": "veo-3",
      "type": "video_generation",
      "status": "available",
      "version": "3.0",
      "max_duration": "60s"
    }
  ]
}
```

## 🔑 API Keys Management

### GET /api/keys
List user's API keys.

**Response:**
```json
[
  {
    "id": "key_123456789",
    "name": "Production Integration",
    "key": "pk_live_abcd...1234",
    "created_at": "2024-01-15T10:00:00Z",
    "last_used": "2024-01-15T15:30:00Z",
    "permissions": ["build", "deploy"],
    "rate_limit": 100
  }
]
```

### POST /api/keys
Create a new API key.

**Request Body:**
```json
{
  "name": "Development Testing",
  "permissions": ["build"],
  "rate_limit": 50
}
```

## 📝 Error Codes Reference

| Code | Description | Resolution |
|------|-------------|------------|
| `INVALID_INPUT` | Request validation failed | Check request format and required fields |
| `RATE_LIMIT_EXCEEDED` | Too many requests | Wait for rate limit reset |
| `BUILD_FAILED` | App building process failed | Check prompt complexity and try again |
| `DEPLOYMENT_FAILED` | Deployment to Cloud Run failed | Verify project configuration |
| `MODEL_UNAVAILABLE` | AI model temporarily unavailable | Try again or use fallback model |
| `INSUFFICIENT_CREDITS` | Account has insufficient credits | Upgrade plan or purchase credits |
| `SESSION_NOT_FOUND` | Build session doesn't exist | Verify session ID |
| `UNAUTHORIZED` | Invalid or expired token | Refresh authentication token |

## 📏 Rate Limits

| Endpoint | Rate Limit | Window |
|----------|------------|--------|
| `/api/nexus-forge/build` | 5 requests | 1 hour |
| `/api/nexus-forge/build/{id}` | 60 requests | 1 minute |
| `/api/nexus-forge/builds` | 100 requests | 1 minute |
| Authentication endpoints | 10 requests | 5 minutes |
| All other endpoints | 1000 requests | 1 hour |

## 🧪 Testing & Examples

### Python SDK Example
```python
import nexusforge

# Initialize client
client = nexusforge.Client(api_key="pk_live_...")

# Start app build
build = client.builds.create(
    prompt="Build a todo app with React and FastAPI",
    config={
        "useAdaptiveThinking": True,
        "enableVideoDemo": True
    }
)

# Monitor progress
for update in client.builds.stream(build.session_id):
    print(f"Phase: {update.phase}, Progress: {update.progress}%")
    
    if update.type == "build_completed":
        print(f"App deployed at: {update.deployment_url}")
        break
```

### JavaScript/Node.js Example
```javascript
const NexusForge = require('nexusforge-sdk');

const client = new NexusForge({
  apiKey: 'pk_live_...',
  baseUrl: 'https://api.nexusforge.example.com'
});

async function buildApp() {
  // Start build
  const build = await client.builds.create({
    prompt: 'Build a real-time chat application',
    config: {
      useAdaptiveThinking: true,
      enableVideoDemo: true
    }
  });
  
  // Connect to WebSocket for updates
  const ws = client.builds.subscribe(build.sessionId);
  
  ws.on('progress_update', (data) => {
    console.log(`${data.phase}: ${data.progress}%`);
  });
  
  ws.on('build_completed', (data) => {
    console.log(`App deployed: ${data.deployment_url}`);
  });
}
```

### cURL Examples
```bash
# Start a build
curl -X POST https://api.nexusforge.example.com/api/nexus-forge/build \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Build a project management tool with task tracking",
    "config": {
      "useAdaptiveThinking": true,
      "enableVideoDemo": true
    }
  }'

# Check build status
curl https://api.nexusforge.example.com/api/nexus-forge/build/550e8400-e29b-41d4-a716-446655440000 \
  -H "Authorization: Bearer $TOKEN"

# List all builds
curl https://api.nexusforge.example.com/api/nexus-forge/builds?limit=10 \
  -H "Authorization: Bearer $TOKEN"
```

---

*This API reference provides comprehensive documentation for integrating with Nexus Forge. For additional support, contact our developer team or visit the community forum.*