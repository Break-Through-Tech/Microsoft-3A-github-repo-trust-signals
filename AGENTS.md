# General Guidelines

- When writing documentation
  - Keep it very concise
  - No emojis or em dashes.
  - Documentation should be written exactly like it is for production-grade, polished projects.
  - Please do not use tables unless asked for or they are absolutely the right choice.
  - DO NOT hard wrap in md files, especially in the middle of a line. Follow the existing structure and prefer semantic line breaks.
- Prefer to ask the user more questions to clarify the user's requests.
- This project will involve working with repos that are potentially malicious. NEVER download or execute the actual contents of the repo.


# Python Development Instructions

- Use `uv` as the package manager and to run scripts.
- `ty` by Astral is used for type checking. Always add appropriate type hints such that the code would pass ty's type check.
- Follow the Google Python Style Guide.
- At this stage of the project, NEVER add imports to __init__.py files. Leave them empty unless absolutely necessary.
- Always prefer pathlib for dealing with files. Use `Path.open` instead of `open`.
- When using pathlib, **always** Use `.parents[i]` syntax to go up directories instead of using `.parent` multiple times.
- When writing tests, use pytest and pytest-asyncio.
- NEVER use `# type: ignore`. It is better to leave the issue and have the user work with you to fix it.
- Don't put types in quotes unless it is absolutely necessary to avoid circular imports and forward references.
- When adding new dependencies, you **must** use `uv add <package>`. AFTER that, update the `pyproject.toml` to follow the convention for versions like the other dependencies.
- To learn about how packages work, you should read from the relevant source code. This is especially important when determining which types to use.
- Make sure the checks pass: `uv run ruff format && uv run ruff check --fix && uv run ty check`


# Key Files

@Challenge-Project-Overview.md

@README.md
