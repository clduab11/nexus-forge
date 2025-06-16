-- Marketplace Tables Migration
-- Creates tables for agent marketplace functionality

-- Agent packages table
CREATE TABLE IF NOT EXISTS agent_packages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    manifest JSONB NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    
    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    published_at TIMESTAMP WITH TIME ZONE,
    
    -- Author info
    author_id UUID NOT NULL REFERENCES auth.users(id),
    author_email VARCHAR(255) NOT NULL,
    organization VARCHAR(255),
    
    -- Package info
    package_url TEXT,
    package_size_bytes BIGINT,
    package_checksum VARCHAR(64),
    
    -- Validation results
    security_report JSONB,
    performance_metrics JSONB,
    
    -- Marketplace metrics
    downloads INTEGER DEFAULT 0,
    stars INTEGER DEFAULT 0,
    rating DECIMAL(2,1),
    rating_count INTEGER DEFAULT 0,
    
    -- Review process
    review_status VARCHAR(50),
    review_notes TEXT,
    reviewed_by UUID REFERENCES auth.users(id),
    reviewed_at TIMESTAMP WITH TIME ZONE,
    
    -- Constraints
    CONSTRAINT valid_status CHECK (status IN ('pending', 'approved', 'rejected', 'deprecated')),
    CONSTRAINT valid_rating CHECK (rating >= 0 AND rating <= 5)
);

-- Indexes for agent packages
CREATE INDEX idx_agent_packages_name ON agent_packages((manifest->>'name'));
CREATE INDEX idx_agent_packages_status ON agent_packages(status);
CREATE INDEX idx_agent_packages_author ON agent_packages(author_id);
CREATE INDEX idx_agent_packages_created ON agent_packages(created_at DESC);
CREATE INDEX idx_agent_packages_downloads ON agent_packages(downloads DESC);
CREATE INDEX idx_agent_packages_rating ON agent_packages(rating DESC NULLS LAST);
CREATE INDEX idx_agent_packages_category ON agent_packages((manifest->>'category'));
CREATE INDEX idx_agent_packages_tags ON agent_packages USING gin((manifest->'tags'));

-- Agent ratings table
CREATE TABLE IF NOT EXISTS agent_ratings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID NOT NULL REFERENCES agent_packages(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES auth.users(id),
    rating INTEGER NOT NULL CHECK (rating >= 1 AND rating <= 5),
    review TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    helpful_count INTEGER DEFAULT 0,
    
    -- Unique constraint: one rating per user per agent
    CONSTRAINT unique_user_agent_rating UNIQUE(agent_id, user_id)
);

-- Indexes for ratings
CREATE INDEX idx_agent_ratings_agent ON agent_ratings(agent_id);
CREATE INDEX idx_agent_ratings_user ON agent_ratings(user_id);
CREATE INDEX idx_agent_ratings_created ON agent_ratings(created_at DESC);

-- Agent installations table
CREATE TABLE IF NOT EXISTS agent_installations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID NOT NULL REFERENCES agent_packages(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES auth.users(id),
    installed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_used_at TIMESTAMP WITH TIME ZONE,
    
    -- Installation metadata
    installation_path TEXT,
    version_installed VARCHAR(50),
    
    -- Unique constraint: track each installation
    CONSTRAINT unique_user_agent_install UNIQUE(agent_id, user_id)
);

-- Indexes for installations
CREATE INDEX idx_installations_agent ON agent_installations(agent_id);
CREATE INDEX idx_installations_user ON agent_installations(user_id);
CREATE INDEX idx_installations_date ON agent_installations(installed_at DESC);

-- Review queue table
CREATE TABLE IF NOT EXISTS review_queue (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID NOT NULL REFERENCES agent_packages(id) ON DELETE CASCADE,
    agent_name VARCHAR(255) NOT NULL,
    agent_version VARCHAR(50) NOT NULL,
    author_id UUID NOT NULL REFERENCES auth.users(id),
    priority VARCHAR(50) DEFAULT 'normal',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    claimed_by UUID REFERENCES auth.users(id),
    claimed_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    
    CONSTRAINT valid_priority CHECK (priority IN ('low', 'normal', 'high', 'urgent'))
);

-- Indexes for review queue
CREATE INDEX idx_review_queue_priority ON review_queue(priority, created_at);
CREATE INDEX idx_review_queue_unclaimed ON review_queue(claimed_by) WHERE claimed_by IS NULL;

-- Agent dependencies table
CREATE TABLE IF NOT EXISTS agent_dependencies (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID NOT NULL REFERENCES agent_packages(id) ON DELETE CASCADE,
    depends_on_id UUID NOT NULL REFERENCES agent_packages(id),
    version_constraint VARCHAR(100),
    
    CONSTRAINT unique_dependency UNIQUE(agent_id, depends_on_id)
);

-- Indexes for dependencies
CREATE INDEX idx_dependencies_agent ON agent_dependencies(agent_id);
CREATE INDEX idx_dependencies_depends_on ON agent_dependencies(depends_on_id);

-- Marketplace statistics table (for caching aggregated stats)
CREATE TABLE IF NOT EXISTS marketplace_stats (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    stat_date DATE NOT NULL DEFAULT CURRENT_DATE,
    total_agents INTEGER DEFAULT 0,
    total_downloads INTEGER DEFAULT 0,
    total_authors INTEGER DEFAULT 0,
    new_agents_today INTEGER DEFAULT 0,
    downloads_today INTEGER DEFAULT 0,
    agents_by_category JSONB,
    popular_tags JSONB,
    trending_agents JSONB,
    
    CONSTRAINT unique_stat_date UNIQUE(stat_date)
);

-- Functions and triggers

-- Function to update agent rating
CREATE OR REPLACE FUNCTION update_agent_rating()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE agent_packages
    SET 
        rating = (
            SELECT AVG(rating)::DECIMAL(2,1)
            FROM agent_ratings
            WHERE agent_id = NEW.agent_id
        ),
        rating_count = (
            SELECT COUNT(*)
            FROM agent_ratings
            WHERE agent_id = NEW.agent_id
        )
    WHERE id = NEW.agent_id;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to update rating on new rating
CREATE TRIGGER update_agent_rating_trigger
AFTER INSERT OR UPDATE ON agent_ratings
FOR EACH ROW
EXECUTE FUNCTION update_agent_rating();

-- Function to increment downloads
CREATE OR REPLACE FUNCTION increment_downloads(p_agent_id UUID)
RETURNS VOID AS $$
BEGIN
    UPDATE agent_packages
    SET downloads = downloads + 1
    WHERE id = p_agent_id;
END;
$$ LANGUAGE plpgsql;

-- Function to update timestamps
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Add update triggers
CREATE TRIGGER update_agent_packages_updated_at
BEFORE UPDATE ON agent_packages
FOR EACH ROW
EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER update_agent_ratings_updated_at
BEFORE UPDATE ON agent_ratings
FOR EACH ROW
EXECUTE FUNCTION update_updated_at();

-- Row Level Security (RLS) policies

-- Enable RLS
ALTER TABLE agent_packages ENABLE ROW LEVEL SECURITY;
ALTER TABLE agent_ratings ENABLE ROW LEVEL SECURITY;
ALTER TABLE agent_installations ENABLE ROW LEVEL SECURITY;
ALTER TABLE review_queue ENABLE ROW LEVEL SECURITY;

-- Agent packages policies
-- Anyone can view approved agents
CREATE POLICY view_approved_agents ON agent_packages
    FOR SELECT
    USING (status = 'approved' OR auth.uid() = author_id);

-- Authors can insert their own agents
CREATE POLICY insert_own_agents ON agent_packages
    FOR INSERT
    WITH CHECK (auth.uid() = author_id);

-- Authors can update their own pending agents
CREATE POLICY update_own_agents ON agent_packages
    FOR UPDATE
    USING (auth.uid() = author_id AND status = 'pending')
    WITH CHECK (auth.uid() = author_id);

-- Ratings policies
-- Anyone can view ratings
CREATE POLICY view_all_ratings ON agent_ratings
    FOR SELECT
    USING (true);

-- Users can insert/update their own ratings
CREATE POLICY manage_own_ratings ON agent_ratings
    FOR ALL
    USING (auth.uid() = user_id)
    WITH CHECK (auth.uid() = user_id);

-- Installation policies
-- Users can view their own installations
CREATE POLICY view_own_installations ON agent_installations
    FOR SELECT
    USING (auth.uid() = user_id);

-- Users can manage their own installations
CREATE POLICY manage_own_installations ON agent_installations
    FOR ALL
    USING (auth.uid() = user_id)
    WITH CHECK (auth.uid() = user_id);

-- Review queue policies (admin only)
-- Admins can view all review items
CREATE POLICY admin_view_review_queue ON review_queue
    FOR SELECT
    USING (
        EXISTS (
            SELECT 1 FROM auth.users
            WHERE id = auth.uid()
            AND raw_user_meta_data->>'is_admin' = 'true'
        )
    );

-- Admins can update review items
CREATE POLICY admin_update_review_queue ON review_queue
    FOR UPDATE
    USING (
        EXISTS (
            SELECT 1 FROM auth.users
            WHERE id = auth.uid()
            AND raw_user_meta_data->>'is_admin' = 'true'
        )
    );

-- Comments
COMMENT ON TABLE agent_packages IS 'Stores agent packages published to the marketplace';
COMMENT ON TABLE agent_ratings IS 'User ratings and reviews for agents';
COMMENT ON TABLE agent_installations IS 'Tracks agent installations by users';
COMMENT ON TABLE review_queue IS 'Queue for agents pending admin review';
COMMENT ON TABLE agent_dependencies IS 'Agent dependency relationships';
COMMENT ON TABLE marketplace_stats IS 'Cached marketplace statistics';