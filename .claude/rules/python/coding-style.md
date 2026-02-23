# Python Coding Style

> This file extends [common/coding-style.md](../common/coding-style.md) with Python specific content.

## Standards

- Follow **PEP 8** conventions
- Follow **Google Python Style Guide**: https://raw.githubusercontent.com/google/styleguide/refs/heads/gh-pages/pyguide.md
- Use **type annotations** on all function signatures
- Use **Google-style docstrings** for all functions and classes

## Docstrings (REQUIRED)

All functions and classes MUST have Google-style docstrings:

```python
def fetch_user(user_id: str, include_inactive: bool = False) -> User | None:
    """Fetch a user by ID from the database.

    Args:
        user_id: The unique identifier of the user.
        include_inactive: Whether to include inactive users. Defaults to False.

    Returns:
        The User object if found, None otherwise.

    Raises:
        DatabaseError: If the database connection fails.
    """
    ...
```

Verify docstring style with:
```bash
ruff check --select D src/
```

## Immutability

Prefer immutable data structures:

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class User:
    name: str
    email: str

from typing import NamedTuple

class Point(NamedTuple):
    x: float
    y: float
```

## Loop Prohibition (CRITICAL)

NEVER use for-loops. Use instead:

```python
# WRONG: for-loop
total = 0
for item in items:
    total += item.price

# CORRECT: List comprehension
total = sum(item.price for item in items)

# CORRECT: itertools
from itertools import chain, groupby

# CORRECT: Numpy/Pandas vectorization
df['total'] = df['quantity'] * df['price']

# CORRECT: map(), filter(), functools.reduce()
from functools import reduce
result = reduce(lambda acc, x: acc + x, items)
```

Rationale: for-loops encourage mutation and imperative thinking. Functional alternatives are more expressive, composable, and less error-prone.

## Package Management

- NEVER use `pip` directly
- ALWAYS use **uv** for package management:
  ```bash
  uv run python script.py    # Run Python scripts
  uv add package-name        # Add dependencies
  uv remove package-name     # Remove dependencies
  uv sync                    # Sync dependencies
  ```
- NEVER edit `pyproject.toml` directly
  - Use `uv add`, `uv remove`, `uv sync` to modify dependencies
  - Let uv manage the file automatically

## Formatting

- **ruff** for formatting, linting, and import sorting (ruff only - no black/isort)

## Reference

See skill: `python-patterns` for comprehensive Python idioms and patterns.
