# Useful Commands Template

This folder contains a template for organizing useful commands in any project.

## Template Structure

```
.useful_commands/
├── useful_commands.txt    # Main commands file (copy to projects)
└── README.md              # This file
```

## Usage

1. Copy `.templates/useful_commands_template/` to your project root
2. Rename to `.useful_commands`
3. Populate `useful_commands.txt` with project-specific commands
4. Update commands as the project evolves

## AI Instructions

When working on a new project:

1. **Check if `.useful_commands/` exists** - If not, create it from template
2. **Add commands as you discover them** - Every time you run a useful command, add it
3. **Organize by category** - Training, testing, deployment, etc.
4. **Keep it current** - Remove outdated commands, update changed ones

## Command Format

```
## CATEGORY NAME

# Description of what the command does
actual-command-to-run

# Another command
another-command --with-flags
```
