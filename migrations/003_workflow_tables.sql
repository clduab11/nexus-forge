-- Workflow Builder Tables Migration
-- Creates tables for visual workflow functionality

-- Workflows table
CREATE TABLE IF NOT EXISTS workflows (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    version VARCHAR(50) DEFAULT '1.0.0',
    
    -- Structure (stored as JSONB for flexibility)
    nodes JSONB DEFAULT '[]'::jsonb,
    connections JSONB DEFAULT '[]'::jsonb,
    variables JSONB DEFAULT '[]'::jsonb,
    triggers JSONB DEFAULT '[]'::jsonb,
    
    -- Metadata
    author_id UUID NOT NULL REFERENCES auth.users(id),
    organization VARCHAR(255),
    tags TEXT[] DEFAULT '{}',
    category VARCHAR(100) DEFAULT 'general',
    
    -- Status
    published BOOLEAN DEFAULT false,
    public BOOLEAN DEFAULT false,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Statistics
    executions INTEGER DEFAULT 0,
    last_executed TIMESTAMP WITH TIME ZONE
);

-- Indexes for workflows
CREATE INDEX idx_workflows_author ON workflows(author_id);
CREATE INDEX idx_workflows_category ON workflows(category);
CREATE INDEX idx_workflows_published ON workflows(published);
CREATE INDEX idx_workflows_public ON workflows(public);
CREATE INDEX idx_workflows_created ON workflows(created_at DESC);
CREATE INDEX idx_workflows_updated ON workflows(updated_at DESC);
CREATE INDEX idx_workflows_tags ON workflows USING gin(tags);
CREATE INDEX idx_workflows_name ON workflows(name);

-- Workflow executions table
CREATE TABLE IF NOT EXISTS workflow_executions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workflow_id UUID NOT NULL REFERENCES workflows(id) ON DELETE CASCADE,
    workflow_version VARCHAR(50) DEFAULT '1.0.0',
    
    -- Execution details
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    trigger_type VARCHAR(100) NOT NULL,
    trigger_data JSONB DEFAULT '{}'::jsonb,
    
    -- Runtime state
    current_node UUID,
    node_states JSONB DEFAULT '{}'::jsonb,
    variables JSONB DEFAULT '{}'::jsonb,
    
    -- Results
    output JSONB,
    error TEXT,
    
    -- Timestamps
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    
    -- Performance metrics
    duration_ms INTEGER,
    nodes_executed INTEGER DEFAULT 0,
    
    -- User context
    user_id UUID REFERENCES auth.users(id),
    tenant_id UUID,
    
    -- Constraints
    CONSTRAINT valid_status CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled', 'paused'))
);

-- Indexes for workflow executions
CREATE INDEX idx_executions_workflow ON workflow_executions(workflow_id);
CREATE INDEX idx_executions_user ON workflow_executions(user_id);
CREATE INDEX idx_executions_status ON workflow_executions(status);
CREATE INDEX idx_executions_started ON workflow_executions(started_at DESC);
CREATE INDEX idx_executions_completed ON workflow_executions(completed_at DESC);
CREATE INDEX idx_executions_trigger ON workflow_executions(trigger_type);

-- Workflow templates table
CREATE TABLE IF NOT EXISTS workflow_templates (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    description TEXT NOT NULL,
    category VARCHAR(100) NOT NULL,
    
    -- Template definition (complete workflow as JSONB)
    workflow JSONB NOT NULL,
    
    -- Configuration schema for template customization
    config_schema JSONB DEFAULT '{}'::jsonb,
    
    -- Metadata
    author VARCHAR(255) NOT NULL,
    tags TEXT[] DEFAULT '{}',
    difficulty VARCHAR(50) DEFAULT 'beginner',
    
    -- Usage statistics
    usage_count INTEGER DEFAULT 0,
    rating DECIMAL(2,1),
    rating_count INTEGER DEFAULT 0,
    
    -- Publishing
    published BOOLEAN DEFAULT false,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT valid_difficulty CHECK (difficulty IN ('beginner', 'intermediate', 'advanced')),
    CONSTRAINT valid_rating CHECK (rating >= 0 AND rating <= 5)
);

-- Indexes for workflow templates
CREATE INDEX idx_templates_category ON workflow_templates(category);
CREATE INDEX idx_templates_published ON workflow_templates(published);
CREATE INDEX idx_templates_usage ON workflow_templates(usage_count DESC);
CREATE INDEX idx_templates_rating ON workflow_templates(rating DESC NULLS LAST);
CREATE INDEX idx_templates_created ON workflow_templates(created_at DESC);
CREATE INDEX idx_templates_tags ON workflow_templates USING gin(tags);
CREATE INDEX idx_templates_difficulty ON workflow_templates(difficulty);

-- Node execution results table (for detailed monitoring)
CREATE TABLE IF NOT EXISTS node_execution_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    execution_id UUID NOT NULL REFERENCES workflow_executions(id) ON DELETE CASCADE,
    node_id UUID NOT NULL,
    
    -- Execution details
    status VARCHAR(50) NOT NULL,
    input_data JSONB DEFAULT '{}'::jsonb,
    output_data JSONB,
    
    -- Timing
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    duration_ms INTEGER,
    
    -- Error handling
    error TEXT,
    retry_count INTEGER DEFAULT 0,
    
    -- Debug info
    debug_info JSONB,
    
    CONSTRAINT valid_node_status CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled'))
);

-- Indexes for node execution results
CREATE INDEX idx_node_results_execution ON node_execution_results(execution_id);
CREATE INDEX idx_node_results_node ON node_execution_results(node_id);
CREATE INDEX idx_node_results_status ON node_execution_results(status);
CREATE INDEX idx_node_results_started ON node_execution_results(started_at);

-- Workflow sharing table
CREATE TABLE IF NOT EXISTS workflow_shares (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workflow_id UUID NOT NULL REFERENCES workflows(id) ON DELETE CASCADE,
    shared_by UUID NOT NULL REFERENCES auth.users(id),
    shared_with UUID REFERENCES auth.users(id),
    organization VARCHAR(255),
    
    -- Permissions
    can_view BOOLEAN DEFAULT true,
    can_edit BOOLEAN DEFAULT false,
    can_execute BOOLEAN DEFAULT true,
    
    -- Sharing details
    shared_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE,
    
    -- Either shared with specific user or organization (not both)
    CONSTRAINT share_target_check CHECK (
        (shared_with IS NOT NULL AND organization IS NULL) OR
        (shared_with IS NULL AND organization IS NOT NULL)
    )
);

-- Indexes for workflow shares
CREATE INDEX idx_shares_workflow ON workflow_shares(workflow_id);
CREATE INDEX idx_shares_shared_with ON workflow_shares(shared_with);
CREATE INDEX idx_shares_organization ON workflow_shares(organization);
CREATE INDEX idx_shares_shared_by ON workflow_shares(shared_by);

-- Workflow comments table
CREATE TABLE IF NOT EXISTS workflow_comments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workflow_id UUID NOT NULL REFERENCES workflows(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES auth.users(id),
    
    comment TEXT NOT NULL,
    parent_id UUID REFERENCES workflow_comments(id),
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- For threaded comments
    thread_depth INTEGER DEFAULT 0
);

-- Indexes for workflow comments
CREATE INDEX idx_comments_workflow ON workflow_comments(workflow_id);
CREATE INDEX idx_comments_user ON workflow_comments(user_id);
CREATE INDEX idx_comments_parent ON workflow_comments(parent_id);
CREATE INDEX idx_comments_created ON workflow_comments(created_at DESC);

-- Functions and Triggers

-- Function to update workflow execution count
CREATE OR REPLACE FUNCTION increment_workflow_executions(p_workflow_id UUID)
RETURNS VOID AS $$
BEGIN
    UPDATE workflows
    SET 
        executions = executions + 1,
        last_executed = NOW()
    WHERE id = p_workflow_id;
END;
$$ LANGUAGE plpgsql;

-- Function to increment template usage
CREATE OR REPLACE FUNCTION increment_template_usage(p_template_id UUID)
RETURNS VOID AS $$
BEGIN
    UPDATE workflow_templates
    SET usage_count = usage_count + 1
    WHERE id = p_template_id;
END;
$$ LANGUAGE plpgsql;

-- Function to calculate execution duration
CREATE OR REPLACE FUNCTION calculate_execution_duration()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.completed_at IS NOT NULL AND NEW.started_at IS NOT NULL THEN
        NEW.duration_ms = EXTRACT(EPOCH FROM (NEW.completed_at - NEW.started_at)) * 1000;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to calculate execution duration
CREATE TRIGGER calculate_execution_duration_trigger
BEFORE UPDATE ON workflow_executions
FOR EACH ROW
WHEN (NEW.completed_at IS NOT NULL AND OLD.completed_at IS NULL)
EXECUTE FUNCTION calculate_execution_duration();

-- Function to update node execution duration
CREATE OR REPLACE FUNCTION calculate_node_duration()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.completed_at IS NOT NULL AND NEW.started_at IS NOT NULL THEN
        NEW.duration_ms = EXTRACT(EPOCH FROM (NEW.completed_at - NEW.started_at)) * 1000;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to calculate node execution duration
CREATE TRIGGER calculate_node_duration_trigger
BEFORE UPDATE ON node_execution_results
FOR EACH ROW
WHEN (NEW.completed_at IS NOT NULL AND OLD.completed_at IS NULL)
EXECUTE FUNCTION calculate_node_duration();

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_workflow_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to update workflow updated_at
CREATE TRIGGER update_workflow_updated_at_trigger
BEFORE UPDATE ON workflows
FOR EACH ROW
EXECUTE FUNCTION update_workflow_updated_at();

-- Trigger to update template updated_at
CREATE TRIGGER update_template_updated_at_trigger
BEFORE UPDATE ON workflow_templates
FOR EACH ROW
EXECUTE FUNCTION update_workflow_updated_at();

-- Row Level Security (RLS) policies

-- Enable RLS
ALTER TABLE workflows ENABLE ROW LEVEL SECURITY;
ALTER TABLE workflow_executions ENABLE ROW LEVEL SECURITY;
ALTER TABLE workflow_templates ENABLE ROW LEVEL SECURITY;
ALTER TABLE node_execution_results ENABLE ROW LEVEL SECURITY;
ALTER TABLE workflow_shares ENABLE ROW LEVEL SECURITY;
ALTER TABLE workflow_comments ENABLE ROW LEVEL SECURITY;

-- Workflows policies
-- Users can view their own workflows and public ones
CREATE POLICY view_workflows ON workflows
    FOR SELECT
    USING (
        author_id = auth.uid() OR 
        public = true OR
        EXISTS (
            SELECT 1 FROM workflow_shares 
            WHERE workflow_id = id 
            AND (shared_with = auth.uid() OR organization = (
                SELECT raw_user_meta_data->>'organization' 
                FROM auth.users 
                WHERE id = auth.uid()
            ))
            AND can_view = true
        )
    );

-- Users can insert their own workflows
CREATE POLICY insert_workflows ON workflows
    FOR INSERT
    WITH CHECK (author_id = auth.uid());

-- Users can update their own workflows
CREATE POLICY update_workflows ON workflows
    FOR UPDATE
    USING (
        author_id = auth.uid() OR
        EXISTS (
            SELECT 1 FROM workflow_shares 
            WHERE workflow_id = id 
            AND shared_with = auth.uid() 
            AND can_edit = true
        )
    )
    WITH CHECK (
        author_id = auth.uid() OR
        EXISTS (
            SELECT 1 FROM workflow_shares 
            WHERE workflow_id = id 
            AND shared_with = auth.uid() 
            AND can_edit = true
        )
    );

-- Users can delete their own workflows
CREATE POLICY delete_workflows ON workflows
    FOR DELETE
    USING (author_id = auth.uid());

-- Workflow executions policies
-- Users can view executions of workflows they have access to
CREATE POLICY view_executions ON workflow_executions
    FOR SELECT
    USING (
        user_id = auth.uid() OR
        EXISTS (
            SELECT 1 FROM workflows w
            WHERE w.id = workflow_id
            AND (
                w.author_id = auth.uid() OR
                w.public = true OR
                EXISTS (
                    SELECT 1 FROM workflow_shares s
                    WHERE s.workflow_id = w.id
                    AND s.shared_with = auth.uid()
                    AND s.can_view = true
                )
            )
        )
    );

-- Users can insert executions for workflows they can execute
CREATE POLICY insert_executions ON workflow_executions
    FOR INSERT
    WITH CHECK (
        user_id = auth.uid() AND
        EXISTS (
            SELECT 1 FROM workflows w
            WHERE w.id = workflow_id
            AND (
                w.author_id = auth.uid() OR
                w.public = true OR
                EXISTS (
                    SELECT 1 FROM workflow_shares s
                    WHERE s.workflow_id = w.id
                    AND s.shared_with = auth.uid()
                    AND s.can_execute = true
                )
            )
        )
    );

-- Templates policies
-- Anyone can view published templates
CREATE POLICY view_templates ON workflow_templates
    FOR SELECT
    USING (published = true);

-- Admins can insert templates
CREATE POLICY insert_templates ON workflow_templates
    FOR INSERT
    WITH CHECK (
        EXISTS (
            SELECT 1 FROM auth.users
            WHERE id = auth.uid()
            AND raw_user_meta_data->>'is_admin' = 'true'
        )
    );

-- Comments policies
-- Users can view comments on workflows they can view
CREATE POLICY view_comments ON workflow_comments
    FOR SELECT
    USING (
        EXISTS (
            SELECT 1 FROM workflows w
            WHERE w.id = workflow_id
            AND (
                w.author_id = auth.uid() OR
                w.public = true OR
                EXISTS (
                    SELECT 1 FROM workflow_shares s
                    WHERE s.workflow_id = w.id
                    AND s.shared_with = auth.uid()
                    AND s.can_view = true
                )
            )
        )
    );

-- Users can add comments to workflows they can view
CREATE POLICY insert_comments ON workflow_comments
    FOR INSERT
    WITH CHECK (
        user_id = auth.uid() AND
        EXISTS (
            SELECT 1 FROM workflows w
            WHERE w.id = workflow_id
            AND (
                w.author_id = auth.uid() OR
                w.public = true OR
                EXISTS (
                    SELECT 1 FROM workflow_shares s
                    WHERE s.workflow_id = w.id
                    AND s.shared_with = auth.uid()
                    AND s.can_view = true
                )
            )
        )
    );

-- Users can update their own comments
CREATE POLICY update_comments ON workflow_comments
    FOR UPDATE
    USING (user_id = auth.uid())
    WITH CHECK (user_id = auth.uid());

-- Views for analytics

-- Workflow execution summary view
CREATE OR REPLACE VIEW workflow_execution_summary AS
SELECT 
    w.id as workflow_id,
    w.name as workflow_name,
    w.author_id,
    COUNT(e.id) as total_executions,
    COUNT(CASE WHEN e.status = 'completed' THEN 1 END) as successful_executions,
    COUNT(CASE WHEN e.status = 'failed' THEN 1 END) as failed_executions,
    AVG(CASE WHEN e.duration_ms IS NOT NULL THEN e.duration_ms END) as avg_duration_ms,
    MAX(e.started_at) as last_execution
FROM workflows w
LEFT JOIN workflow_executions e ON w.id = e.workflow_id
GROUP BY w.id, w.name, w.author_id;

-- Popular templates view
CREATE OR REPLACE VIEW popular_templates AS
SELECT 
    t.*,
    COALESCE(t.usage_count, 0) as usage_count,
    COALESCE(t.rating, 0) as rating
FROM workflow_templates t
WHERE t.published = true
ORDER BY t.usage_count DESC, t.rating DESC;

-- Comments on workflows
COMMENT ON TABLE workflows IS 'Visual workflows created by users';
COMMENT ON TABLE workflow_executions IS 'Execution instances of workflows';
COMMENT ON TABLE workflow_templates IS 'Reusable workflow templates';
COMMENT ON TABLE node_execution_results IS 'Detailed execution results for individual nodes';
COMMENT ON TABLE workflow_shares IS 'Workflow sharing permissions';
COMMENT ON TABLE workflow_comments IS 'User comments on workflows';