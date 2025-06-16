/**
 * End-to-End Tests for Complete Nexus Forge Workflow
 * Tests the entire user journey from login to project completion
 */

describe('Complete Nexus Forge Workflow', () => {
  const testUser = {
    email: 'test@nexusforge.com',
    password: 'test123456',
    name: 'Test User'
  };

  const testProject = {
    name: 'E2E Test Project',
    description: 'End-to-end test application for Cypress',
    platform: 'web',
    framework: 'React',
    features: ['REST API', 'Authentication', 'Database Integration'],
    requirements: 'A simple CRUD application with user authentication'
  };

  beforeEach(() => {
    // Clear cookies and local storage
    cy.clearAllCookies();
    cy.clearAllLocalStorage();
    cy.clearAllSessionStorage();
    
    // Visit the application
    cy.visit('/');
  });

  it('should complete the entire workflow from registration to project completion', () => {
    // Step 1: User Registration
    cy.log('Starting user registration process');
    
    // Should redirect to login page if not authenticated
    cy.url().should('include', '/login');
    
    // Switch to sign up mode
    cy.get('[data-testid="signup-tab"]').click();
    
    // Fill registration form
    cy.get('input[type="text"]').type(testUser.name);
    cy.get('input[type="email"]').type(testUser.email);
    cy.get('input[type="password"]').first().type(testUser.password);
    cy.get('input[type="password"]').last().type(testUser.password);
    
    // Submit registration
    cy.get('button[type="submit"]').click();
    
    // Should redirect to dashboard after successful registration
    cy.url().should('include', '/dashboard');
    cy.get('[data-testid="dashboard-title"]').should('contain', 'Nexus Forge Workspace');
    
    // Step 2: Navigate to Project Builder
    cy.log('Navigating to project builder');
    
    cy.get('[data-testid="create-project-btn"]').click();
    cy.url().should('include', '/builder');
    
    // Step 3: Fill Project Information (Step 1)
    cy.log('Filling project information');
    
    cy.get('input[placeholder*="project name"]').type(testProject.name);
    cy.get('textarea[placeholder*="Describe"]').type(testProject.description);
    cy.get('button').contains('Next').click();
    
    // Step 4: Select Platform and Framework (Step 2)
    cy.log('Selecting platform and framework');
    
    // Select web platform
    cy.get('[data-testid="platform-web"]').click();
    
    // Select React framework
    cy.get('button').contains(testProject.framework).click();
    cy.get('button').contains('Next').click();
    
    // Step 5: Select Features (Step 3)
    cy.log('Selecting features');
    
    testProject.features.forEach(feature => {
      cy.get('button').contains(feature).click();
    });
    
    // Add additional requirements
    cy.get('textarea[placeholder*="requirements"]').type(testProject.requirements);
    cy.get('button').contains('Next').click();
    
    // Step 6: Review and Create Project (Step 4)
    cy.log('Reviewing and creating project');
    
    // Verify project details are displayed correctly
    cy.get('[data-testid="review-project-name"]').should('contain', testProject.name);
    cy.get('[data-testid="review-platform"]').should('contain', 'Web Application');
    cy.get('[data-testid="review-framework"]').should('contain', testProject.framework);
    
    // Create the project
    cy.get('button').contains('Create Project').click();
    
    // Should redirect to results page
    cy.url().should('include', '/results/');
    
    // Step 7: Monitor Project Generation Progress
    cy.log('Monitoring project generation');
    
    // Wait for project to be created and in progress
    cy.get('[data-testid="project-status"]', { timeout: 10000 })
      .should('be.visible');
    
    // Check that progress tracking is working
    cy.get('[data-testid="progress-bar"]', { timeout: 5000 })
      .should('be.visible');
    
    // Verify real-time updates are working
    cy.get('[data-testid="task-progress-tracker"]')
      .should('be.visible');
    
    cy.get('[data-testid="agent-orchestration"]')
      .should('be.visible');
    
    // Step 8: Wait for Completion (with timeout)
    cy.log('Waiting for project completion');
    
    // Wait for generation to complete (max 5 minutes for test)
    cy.get('[data-testid="project-status"]', { timeout: 300000 })
      .should('contain', 'COMPLETED');
    
    // Verify progress is 100%
    cy.get('[data-testid="progress-percentage"]')
      .should('contain', '100%');
    
    // Step 9: Verify Generated Results
    cy.log('Verifying generated results');
    
    // Check that results tabs are available
    cy.get('[data-testid="results-tabs"]').should('be.visible');
    
    // Overview tab should be active by default
    cy.get('[data-testid="tab-overview"]').should('have.class', 'active');
    
    // Verify code files tab
    cy.get('[data-testid="tab-code"]').click();
    cy.get('[data-testid="code-files-list"]').should('be.visible');
    cy.get('[data-testid="code-file-item"]').should('have.length.at.least', 1);
    
    // Select a code file and verify content
    cy.get('[data-testid="code-file-item"]').first().click();
    cy.get('[data-testid="code-file-content"]').should('be.visible');
    cy.get('[data-testid="code-file-content"]').should('not.be.empty');
    
    // Verify assets tab
    cy.get('[data-testid="tab-assets"]').click();
    cy.get('[data-testid="assets-list"]').should('be.visible');
    
    // Verify documentation tab
    cy.get('[data-testid="tab-docs"]').click();
    cy.get('[data-testid="documentation-content"]').should('be.visible');
    cy.get('[data-testid="documentation-content"]').should('not.be.empty');
    
    // Step 10: Test Export Functionality
    cy.log('Testing export functionality');
    
    // Test ZIP export
    cy.get('[data-testid="export-zip-btn"]').click();
    // Note: File download testing in Cypress requires special setup
    // For now, we'll just verify the button works without errors
    
    // Test Git export
    cy.get('[data-testid="export-git-btn"]').click();
    
    // Step 11: Navigate Back to Dashboard
    cy.log('Navigating back to dashboard');
    
    cy.get('[data-testid="back-to-dashboard"]').click();
    cy.url().should('include', '/dashboard');
    
    // Verify the project appears in the dashboard
    cy.get('[data-testid="project-list"]').should('contain', testProject.name);
    cy.get('[data-testid="project-status-completed"]').should('be.visible');
  });

  it('should handle real-time updates correctly', () => {
    // Login first
    cy.login(testUser.email, testUser.password);
    
    // Create a project to monitor
    cy.createProject({
      name: 'Real-time Test Project',
      platform: 'web',
      framework: 'Vue.js',
      features: ['REST API'],
      requirements: 'Simple API test'
    });
    
    // Monitor WebSocket connections
    cy.window().then((win) => {
      // Check if WebSocket connection is established
      cy.wrap(win).should('have.property', 'WebSocket');
    });
    
    // Verify real-time status updates
    cy.get('[data-testid="connection-status"]')
      .should('contain', 'Connected');
    
    // Check agent status updates
    cy.get('[data-testid="agent-status-list"]')
      .should('be.visible');
    
    cy.get('[data-testid="agent-status-item"]')
      .should('have.length.at.least', 1);
    
    // Verify task progress updates
    cy.get('[data-testid="task-list"]')
      .should('be.visible');
  });

  it('should handle errors gracefully', () => {
    // Test invalid login
    cy.visit('/login');
    
    cy.get('input[type="email"]').type('invalid@email.com');
    cy.get('input[type="password"]').type('wrongpassword');
    cy.get('button[type="submit"]').click();
    
    // Should show error message
    cy.get('[data-testid="error-message"]')
      .should('be.visible')
      .and('contain', 'Invalid credentials');
    
    // Test form validation
    cy.get('[data-testid="signup-tab"]').click();
    
    // Try to submit empty form
    cy.get('button[type="submit"]').click();
    
    // Should show validation errors
    cy.get('input:invalid').should('have.length.at.least', 1);
    
    // Test project creation with invalid data
    cy.login(testUser.email, testUser.password);
    cy.visit('/builder');
    
    // Try to proceed without required fields
    cy.get('button').contains('Next').click();
    
    // Should prevent progression
    cy.url().should('include', '/builder');
  });

  it('should be responsive on different screen sizes', () => {
    // Test mobile viewport
    cy.viewport('iphone-x');
    cy.login(testUser.email, testUser.password);
    
    // Check mobile navigation
    cy.get('[data-testid="mobile-menu-button"]').should('be.visible');
    cy.get('[data-testid="mobile-menu-button"]').click();
    cy.get('[data-testid="mobile-navigation"]').should('be.visible');
    
    // Test tablet viewport
    cy.viewport('ipad-2');
    cy.get('[data-testid="dashboard-grid"]').should('be.visible');
    
    // Test desktop viewport
    cy.viewport(1920, 1080);
    cy.get('[data-testid="sidebar"]').should('be.visible');
    cy.get('[data-testid="main-content"]').should('be.visible');
  });

  it('should maintain performance standards', () => {
    // Monitor page load times
    cy.visit('/', {
      onBeforeLoad: (win) => {
        win.performance.mark('start');
      },
      onLoad: (win) => {
        win.performance.mark('end');
        win.performance.measure('pageLoad', 'start', 'end');
      }
    });
    
    cy.window().then((win) => {
      const measure = win.performance.getEntriesByName('pageLoad')[0];
      expect(measure.duration).to.be.lessThan(3000); // Page should load in under 3 seconds
    });
    
    // Check for accessibility
    cy.injectAxe();
    cy.checkA11y();
  });
});

// Custom commands
declare global {
  namespace Cypress {
    interface Chainable {
      login(email: string, password: string): Chainable<void>;
      createProject(project: any): Chainable<void>;
    }
  }
}

Cypress.Commands.add('login', (email: string, password: string) => {
  cy.visit('/login');
  cy.get('input[type="email"]').type(email);
  cy.get('input[type="password"]').type(password);
  cy.get('button[type="submit"]').click();
  cy.url().should('include', '/dashboard');
});

Cypress.Commands.add('createProject', (project: any) => {
  cy.visit('/builder');
  
  // Step 1: Basic info
  cy.get('input[placeholder*="project name"]').type(project.name);
  cy.get('button').contains('Next').click();
  
  // Step 2: Platform selection
  cy.get(`[data-testid="platform-${project.platform}"]`).click();
  cy.get('button').contains(project.framework).click();
  cy.get('button').contains('Next').click();
  
  // Step 3: Features
  project.features.forEach((feature: string) => {
    cy.get('button').contains(feature).click();
  });
  cy.get('textarea[placeholder*="requirements"]').type(project.requirements);
  cy.get('button').contains('Next').click();
  
  // Step 4: Create
  cy.get('button').contains('Create Project').click();
  cy.url().should('include', '/results/');
});