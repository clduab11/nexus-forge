# 🎨 Visual Workflow UI/UX Design Document

## Overview

This document outlines the visual design and user experience for the Nexus Forge Visual Workflow Builder, including mockups, interaction patterns, and responsive design considerations.

## 1. UI Layout Structure

### 1.1 Main Editor Layout
```
┌─────────────────────────────────────────────────────────────────┐
│                          Toolbar                                 │
├─────────────┬─────────────────────────────────────┬────────────┤
│             │                                       │            │
│   Node      │         Canvas Area                  │  Property  │
│  Palette    │     (Drag & Drop Zone)               │   Panel    │
│             │                                       │            │
│  ┌─────┐    │    ┌─────┐      ┌─────┐            │  Settings  │
│  │Agent│    │    │Node1│ ====> │Node2│            │            │
│  └─────┘    │    └─────┘      └─────┘            │            │
│             │                                       │            │
│  ┌─────┐    │         ┌─────┐                     │            │
│  │Flow │    │         │Node3│                     │            │
│  └─────┘    │         └─────┘                     │            │
│             │                                       │            │
└─────────────┴─────────────────────────────────────┴────────────┘
│                        Status Bar                                │
└─────────────────────────────────────────────────────────────────┘
```

## 2. Component Visual Design

### 2.1 Node Design

#### Agent Node
```
┌─────────────────────────┐
│ 🧠 Starri AI        [x] │  <- Header with icon & close
├─────────────────────────┤
│ Master orchestrator     │  <- Description
│                         │
│ • requirements      ○   │  <- Input port (left)
│                         │
│              specs  ○   │  <- Output port (right)
└─────────────────────────┘
```

#### Visual States
- **Default**: Light background, subtle shadow
- **Hover**: Elevated shadow, slight scale
- **Selected**: Blue border, highlighted background
- **Executing**: Pulsing animation, progress bar
- **Error**: Red border, error icon

### 2.2 Connection Design

#### Bezier Curve Connections
```
Source ○ ~~~~~~~~~~~~~ ○ Target
         \           /
          \         /
           \_______/
```

#### Connection States
- **Default**: Gray line, 2px width
- **Hover**: Blue highlight, 3px width
- **Selected**: Blue color with control points
- **Data Flow**: Green with arrow
- **Control Flow**: Purple with different dash pattern
- **Error**: Red with warning icon

## 3. Interaction Patterns

### 3.1 Drag and Drop Flow

1. **From Palette**:
   ```
   User drags from palette → Ghost node follows cursor → 
   Drop on canvas → Node created at position
   ```

2. **Node Movement**:
   ```
   Click and drag node → Node follows with grid snap → 
   Release to place → Connections update smoothly
   ```

3. **Connection Creation**:
   ```
   Click output port → Connection line follows cursor → 
   Click input port → Connection created with animation
   ```

### 3.2 Gesture Support

- **Pan Canvas**: Middle mouse drag or Space + drag
- **Zoom**: Scroll wheel or pinch gesture
- **Multi-select**: Ctrl/Cmd + click or drag selection box
- **Context Menu**: Right-click on nodes/connections

## 4. Visual Feedback System

### 4.1 Execution Visualization

```
┌─────────────────────────┐
│ ⚡ Executing...     75% │  <- Progress indicator
├─────────────────────────┤
│ ████████████░░░░░░░    │  <- Progress bar
│                         │
│ Duration: 2.5s          │  <- Metrics
└─────────────────────────┘
```

### 4.2 Real-time Updates

- **Node Glow**: Active nodes have pulsing glow effect
- **Data Flow**: Animated particles along connections
- **Status Icons**: ✓ Complete, ⚠ Warning, ✗ Error
- **Progress Ripple**: Expanding ripple effect on completion

## 5. Color Palette

### 5.1 Agent Colors
```
Starri AI    - Purple  (#8B5CF6)
Jules Coder  - Blue    (#3B82F6)
Gemini AI    - Green   (#10B981)
Researcher   - Yellow  (#F59E0B)
Developer    - Indigo  (#6366F1)
Designer     - Pink    (#EC4899)
Tester       - Red     (#EF4444)
Analyst      - Cyan    (#06B6D4)
Optimizer    - Orange  (#F97316)
```

### 5.2 UI Colors
```
Background   - Gray 50  (#F9FAFB)
Canvas       - White    (#FFFFFF)
Grid         - Gray 200 (#E5E7EB)
Border       - Gray 300 (#D1D5DB)
Text Primary - Gray 900 (#111827)
Text Second  - Gray 600 (#4B5563)
Primary      - Blue 600 (#2563EB)
Success      - Green    (#10B981)
Warning      - Amber    (#F59E0B)
Error        - Red      (#EF4444)
```

## 6. Responsive Design

### 6.1 Desktop (1920x1080)
- Full 3-column layout
- All panels visible
- Optimal zoom level

### 6.2 Tablet (1024x768)
- Collapsible side panels
- Touch-optimized controls
- Larger hit areas for connections

### 6.3 Mobile (375x812)
- View-only mode
- Pan and zoom only
- Simplified node display

## 7. Accessibility Features

### 7.1 Keyboard Navigation
- **Tab**: Navigate between nodes
- **Enter**: Edit selected node
- **Delete**: Remove selected items
- **Arrows**: Move selected nodes
- **Ctrl+Z/Y**: Undo/Redo

### 7.2 Screen Reader Support
- ARIA labels on all interactive elements
- Role descriptions for workflow components
- Status announcements for execution

### 7.3 Visual Accessibility
- High contrast mode support
- Colorblind-friendly palette options
- Adjustable text size
- Focus indicators

## 8. Animation and Transitions

### 8.1 Micro-interactions
```javascript
// Node hover effect
transform: scale(1.02);
transition: all 0.2s ease;
box-shadow: 0 4px 12px rgba(0,0,0,0.1);

// Connection creation
@keyframes dash {
  to { stroke-dashoffset: -10; }
}
stroke-dasharray: 5,5;
animation: dash 0.5s linear infinite;

// Execution pulse
@keyframes pulse {
  0% { opacity: 1; }
  50% { opacity: 0.6; }
  100% { opacity: 1; }
}
animation: pulse 2s ease-in-out infinite;
```

### 8.2 Page Transitions
- Smooth fade between views
- Slide animations for panels
- Spring physics for drag interactions

## 9. Empty States

### 9.1 Empty Canvas
```
┌─────────────────────────────────┐
│                                 │
│      🎯                         │
│                                 │
│   Start Building Your Workflow  │
│                                 │
│  Drag nodes from the palette    │
│  or choose a template to begin  │
│                                 │
│     [Choose Template]           │
│                                 │
└─────────────────────────────────┘
```

### 9.2 No Connections
```
💡 Tip: Click an output port and then an input port to create connections
```

## 10. Error States

### 10.1 Invalid Connection
```
┌─────────────────────────┐
│ ⚠️ Invalid Connection   │
│                         │
│ Cannot connect:         │
│ Output type: Object     │
│ Input type: String      │
│                         │
│ [Dismiss]               │
└─────────────────────────┘
```

### 10.2 Execution Error
```
┌─────────────────────────┐
│ ❌ Execution Failed     │
│                         │
│ Node: Data Processor    │
│ Error: Invalid input    │
│                         │
│ [View Details] [Retry]  │
└─────────────────────────┘
```

## 11. Mobile Gestures

- **Pinch**: Zoom in/out
- **Two-finger drag**: Pan canvas
- **Long press**: Context menu
- **Double tap**: Center and zoom node
- **Swipe**: Navigate between panels

## 12. Performance Optimizations

### 12.1 Rendering
- Virtual scrolling for large node palettes
- Canvas rendering with WebGL for complex workflows
- Lazy loading of node details
- Debounced connection updates

### 12.2 Interaction
- 60 FPS animations
- Immediate visual feedback
- Progressive rendering for large workflows
- Cached node previews

This design system ensures a professional, intuitive, and performant visual workflow building experience that aligns with modern UI/UX best practices while maintaining the unique identity of Nexus Forge.