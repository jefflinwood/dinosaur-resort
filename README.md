# AI Agent Dinosaur Simulator

A SimCity-style simulation game featuring a dinosaur resort where AI agents with distinct personalities react to player-triggered events using the ag2 framework and OpenAI.

## Project Structure

```
dinosaur-resort/
├── models/                 # Core data models and configurations
│   ├── __init__.py
│   ├── core.py            # Agent, Event, MetricsSnapshot, SimulationState models
│   ├── enums.py           # Enums for roles, states, event types, etc.
│   └── config.py          # Configuration classes
├── managers/              # System managers (to be implemented)
│   └── __init__.py
├── agents/                # AI agent implementations (to be implemented)
│   └── __init__.py
├── ui/                    # Streamlit UI components (to be implemented)
│   └── __init__.py
├── tests/                 # Unit tests
│   ├── __init__.py
│   └── test_models.py     # Tests for core data models
├── .kiro/                 # Kiro spec files
│   └── specs/
│       └── ai-agent-dinosaur-simulator/
│           ├── requirements.md
│           ├── design.md
│           └── tasks.md
├── venv/                  # Virtual environment
├── main.py                # Main application entry point
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Setup

1. Create and activate virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run tests:
```bash
python -m pytest tests/ -v
```

4. Run the application (placeholder):
```bash
streamlit run main.py
```

## Core Components

### Data Models
- **Agent**: Represents AI agents (staff, visitors, dinosaurs) with personalities and capabilities
- **Event**: Represents events that occur in the simulation (escapes, emergencies, etc.)
- **MetricsSnapshot**: Tracks resort performance metrics (satisfaction, happiness, efficiency)
- **SimulationState**: Maintains the current state of the simulation

### Enums
- **AgentRole**: Defines agent types (park ranger, veterinarian, tourist, dinosaur, etc.)
- **EventType**: Defines event categories (escapes, emergencies, facility issues, weather)
- **ResolutionStatus**: Tracks event resolution progress

### Configuration
- **AgentConfig**: Configuration for initializing different types of agents
- **SimulationConfig**: General simulation settings
- **OpenAIConfig**: OpenAI API configuration
- **AG2Config**: ag2 framework configuration

## Development Status

✅ **Task 1 Complete**: Project structure and core data models implemented
- Directory structure created
- Core data models with validation and serialization
- Configuration classes and enums
- Comprehensive unit tests (19 tests passing)
- Requirements.txt with all dependencies

## Next Steps

The next tasks will implement:
- OpenAI and ag2 integration
- Metrics tracking system
- Event management system
- Agent system with personalities
- Streamlit dashboard interface

## Dependencies

- **streamlit**: Web UI framework
- **pyautogen**: ag2 framework for multi-agent orchestration
- **openai**: OpenAI API integration
- **pytest**: Testing framework
- **pydantic**: Data validation
- **python-dotenv**: Environment variable management