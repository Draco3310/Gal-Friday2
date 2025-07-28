# Ruff Version Alignment Rollback Plan

## Overview
This document provides step-by-step procedures to revert all changes made during the ruff version alignment project if issues arise.

## Change Inventory
All changes are being made in the `fix/ruff-version-alignment` branch for safe testing.

### Configuration Files Modified
1. **pyproject.toml** - Line 422: `ruff>=0.1.0` → `ruff>=0.12.0,<0.13.0`
2. **requirements-dev.txt** - Line 384: `ruff==0.8.6` → `ruff==0.12.5`
3. **.pre-commit-config.yaml** - Already at v0.12.5 (no change needed)

### New Files Created
1. **scripts/check_ruff_version.py** - Version validation script
2. **RUFF_MIGRATION_STRATEGY.md** - Migration documentation
3. **ROLLBACK_PLAN.md** - This file

### Modified Files
1. **Makefile** - Added version check to lint targets
2. **CLAUDE.md** - Updated with version requirements
3. **Multiple Python files** - Linting error fixes

## Rollback Procedures

### Emergency Rollback (If Branch Fails)
If the testing branch becomes unusable:
```bash
# Switch back to main branch
git checkout main

# Delete problematic branch
git branch -D fix/ruff-version-alignment

# Restart with fresh branch if needed
git checkout -b fix/ruff-version-alignment-v2
```

### Configuration Rollback
If version alignment causes issues, revert configurations:

#### 1. Revert pyproject.toml
```bash
# Edit pyproject.toml line 422
# Change: ruff>=0.12.0,<0.13.0
# Back to: ruff>=0.1.0
```

#### 2. Revert requirements-dev.txt
```bash
# Edit requirements-dev.txt line 384
# Change: ruff==0.12.5
# Back to: ruff==0.8.6
```

#### 3. Revert pre-commit configuration
```bash
# Edit .pre-commit-config.yaml
# Change rev: v0.12.5
# Back to: rev: v0.8.6
```

#### 4. Update virtual environment
```bash
pip install -r requirements-dev.txt
```

### Code Changes Rollback
If linting fixes cause issues:

#### 1. Revert specific commits
```bash
# View commit history for selective reversion
git log --oneline

# Revert specific commits (replace HASH with actual commit)
git revert HASH

# Or reset to specific point
git reset --hard HASH
```

#### 2. Revert all code changes, keep config
```bash
# Checkout only Python files from main
git checkout main -- gal_friday/ tests/

# Keep configuration changes
# Manual review of other files
```

### Infrastructure Rollback
Remove new validation infrastructure:

#### 1. Remove validation script
```bash
rm scripts/check_ruff_version.py
```

#### 2. Revert Makefile changes
```bash
# Edit Makefile
# Remove ruff-version-check target
# Remove version check from lint dependencies
```

#### 3. Revert CLAUDE.md updates
```bash
# Edit CLAUDE.md
# Remove version requirement documentation
```

## Pre-Rollback Checklist
Before rolling back, verify:
- [ ] Nature of the issue (configuration vs code vs infrastructure)
- [ ] Impact scope (local dev vs CI vs team)
- [ ] Alternative solutions attempted
- [ ] Stakeholder notification if affecting team

## Rollback Validation
After rollback:
- [ ] Local `ruff --version` shows expected version
- [ ] `ruff check` works without errors
- [ ] Pre-commit hooks function correctly
- [ ] CI pipeline passes
- [ ] Test suite executes successfully
- [ ] Team members can continue development

## Recovery Procedures
If rollback itself fails:

#### 1. Hard reset to main
```bash
git checkout main
git reset --hard origin/main
git clean -fdx
```

#### 2. Fresh virtual environment
```bash
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
```

#### 3. Pre-commit reinstall
```bash
pre-commit clean
pre-commit install
```

## Prevention Measures
- All changes in isolated branch
- Systematic testing before main integration
- Incremental commits for granular rollback
- Documentation of all modifications
- Team communication of rollback procedures

## Emergency Contacts
- Technical lead notification required for major rollbacks
- Team notification via standard channels
- Documentation update in project management system

---

**Note**: This rollback plan should be reviewed and approved before beginning implementation. Keep this document updated as changes are made.
