# GEMINI.md - Project Mandates

This document defines the foundational mandates and technical standards for the **All About LLM** project. These instructions take precedence over general workflows.

## Project Overview
A comprehensive documentation repository focused on LLM technologies, serving as a knowledge base for developers and researchers.

- **Primary Language**: Korean (한국어)
- **Tech Stack**: Jekyll (with `just-the-docs` theme), Markdown, Python (for examples and tools).

## Documentation Standards (Markdown & Jekyll)

### Content Structure
- All core documentation resides in the `docs/` directory.
- Each major topic should have its own subdirectory (e.g., `docs/RAG/`, `docs/agent/`).
- Every subdirectory must contain an `index.md` serving as the parent page.

### Jekyll Front Matter
- All Markdown files must include proper YAML front matter for the `just-the-docs` theme.
- **Root Pages (`docs/*/index.md`)**:
  ```yaml
  ---
  layout: default
  title: [Topic Name]
  has_children: true
  nav_order: [Number]
  ---
  ```
- **Child Pages**:
  ```yaml
  ---
  layout: default
  title: [Page Title]
  parent: [Parent Topic Name]
  nav_order: [Number]
  ---
  ```

### Writing Style
- Use clear, professional Korean.
- Use Mermaid for diagrams (supported via Jekyll configuration).
- Ensure relative links are used for internal navigation (e.g., `[Link](./other-page.md)`).

## Technical Standards

### Python Development
- **Formatting**: Strictly follow the [Black](https://black.readthedocs.io/) formatter.
  - **Line Length**: 88 characters (as specified in `pyproject.toml`).
- **Imports**: Use `isort` for sorting imports.
- **Verification**: Always run `black .` before finalizing changes to Python files.

### Repository Health
- **Pre-commit**: Respect the rules defined in `.pre-commit-config.yaml`.
- **Naming**: Use descriptive, snake_case filenames for Python scripts and kebab-case for Markdown files (except `index.md`).

## Workflow Mandates
1.  **Surgical Edits**: When updating documentation, maintain existing navigation (`nav_order`) unless a reorganization is explicitly requested.
2.  **Validation**: After adding new documentation, verify that the front matter correctly aligns with the `just-the-docs` hierarchy to avoid broken navigation.
3.  **Code Examples**: Ensure Python examples are functional and follow the project's formatting rules.
