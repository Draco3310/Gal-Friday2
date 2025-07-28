# Ruff Version Alignment Migration Strategy

## Overview
Migration from ruff v0.8.6 to v0.12.5 revealed ~80 linting errors that were previously silently ignored. This document outlines a systematic approach to resolve these issues.

## Error Analysis (From Current Scan)
Based on `ruff check gal_friday/ tests/ --statistics`:

### Auto-fixable (High Priority - Phase 3A)
- **16 E501** (line-too-long) - Automatic formatting fixes
- **12 ERA001** (commented-out-code) - Remove dead code automatically
- **2 E702** (multiple-statements-on-one-line-semicolon) - Auto-format

**Strategy**: Run `ruff check --fix --unsafe` to resolve ~30 errors automatically

### Simple Manual (High Priority - Phase 3B)
- **12 PT011** (pytest-raises-too-broad) - Add specific exception matching
- **6 FBT003** (boolean-positional-value-in-call) - Add keyword arguments
- **2 SIM117** (multiple-with-statements) - Combine with statements
- **1 T201** (print) - Replace with logging calls

**Strategy**: Systematic fixes, ~20 errors, 2-3 hours

### Medium Complexity (Medium Priority - Phase 3C)
- **13 PLC0415** (import-outside-top-level) - Move imports or add # noqa
- **2 F811** (redefined-while-unused) - Resolve variable naming conflicts
- **2 PGH003** (blanket-type-ignore) - Add specific type ignore comments
- **1 TC001** (typing-only-first-party-import) - Move to TYPE_CHECKING block

**Strategy**: Case-by-case analysis, ~18 errors, 3-4 hours

### Complex/Architectural (Lower Priority - Phase 3D)
- **2 PLR0915** (too-many-statements) - Function decomposition
- **1 C901** (complex-structure) - Complexity reduction
- **1 F821** (undefined-name) - Variable resolution

**Strategy**: Evaluate vs selective # noqa, ~4 errors, 2-4 hours

## Implementation Phases

### Phase 3A: Auto-fixes (30 minutes)
```bash
ruff check --fix --unsafe gal_friday/ tests/
```
Expected: ~30 errors resolved automatically

### Phase 3B: Simple Manual (2-3 hours)
Focus on high-value, low-effort fixes:
1. PT011: Add specific exception types to pytest.raises()
2. FBT003: Convert positional booleans to keyword arguments
3. T201: Replace print() with appropriate logging
4. SIM117: Combine multiple with statements

### Phase 3C: Medium Complexity (3-4 hours)
Systematic resolution of import and type issues:
1. PLC0415: Evaluate import placement, move to top-level where appropriate
2. F811: Resolve variable redefinition conflicts
3. TC001: Move type-only imports to TYPE_CHECKING blocks
4. PGH003: Replace blanket type ignores with specific codes

### Phase 3D: Complex Issues (2-4 hours or selective ignoring)
For complex structural issues:
1. Evaluate if refactoring provides business value
2. If yes: decompose functions, reduce complexity
3. If no: Add selective # noqa with justification comments

## Rollback Plan
1. **Branch Protection**: All work in `fix/ruff-version-alignment` branch
2. **Commit Strategy**: Logical groupings for easy reversion
3. **Configuration Rollback**: Revert pyproject.toml and requirements-dev.txt
4. **Pre-commit Revert**: Downgrade .pre-commit-config.yaml to v0.8.6

## Quality Gates
- [ ] All auto-fixable errors resolved
- [ ] Test suite passes after each phase
- [ ] No functional regressions introduced
- [ ] Code review for manual fixes
- [ ] Final error count: 0 or justified # noqa comments

## Success Criteria
- Local `ruff check` reports same error count as pre-commit
- All high-value errors resolved
- Complex issues either fixed or documented with # noqa
- No impact on functionality or performance
