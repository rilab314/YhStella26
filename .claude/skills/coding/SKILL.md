---
name: coding
description: Apply when writing new code, refactoring existing code, or reviewing code for style issues. Enforces OOP, length limits, naming, and style rules.
version: 1.0.0
---

# Coding Standards

## Core Principles

1. **OOP first** — single responsibility per class and function
2. **No duplication** — repeated multi-line patterns → function; repeated single lines → loop
3. **Readability** — descriptive names over short ones

## Length Limits (recommended ~ hard max)

| Unit | Lines |
|------|-------|
| Module | 200 ~ 400 |
| Class | 100 ~ 200 |
| Function | 20 ~ 30 |
| if/for body | 10 ~ 20 |
| Line width | ≤ 100 chars |

## Naming

- **Noun**: modules, classes, variables (`map_builder`, `lane_mask`)
- **Verb**: functions (`build_map`, `detect_lanes`)
- Prefer descriptive; use consonant abbreviation only when >10 chars
- Short variable names OK when surrounding scope provides clear context

## Style

- **Call order**: arrange functions in depth-first pre-order of the call tree — a caller is immediately followed by its callees before the next sibling appears (e.g. `f1, f2, f4, f5, f3, f6, f7` for a tree where f1→{f2,f3}, f2→{f4,f5}, f3→{f6,f7})
- No bare logic at module level — only imports, constants, `if __name__ == "__main__"`
- Constants → `config.py` or class static members; minimize magic values in logic
- PEP 8 spacing: 1 blank line between methods, 2 lines between top-level definitions
- Bulk data processing → numpy vectorized ops over explicit `for` loops
