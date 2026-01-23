# Claude Self-Reflection Setup

This directory contains files to help Claude (and developers) align with repo goals.

## Files

### SELF_REFLECTION.md
Core framework defining:
- Repo goals and constraints
- Quick reference (DO/DON'T)
- Code patterns
- Pre-change checklist
- Red flags to avoid

**Read this before making any code changes.**

### pre-commit-hook
Git pre-commit hook that validates:
- No hard-coded customer/platform names
- No hard-coded file paths
- Warnings about synthetic data claims
- Warnings about unrealistic "optimal" claims
- CI workflow reminders

### pre-push-hook
Git pre-push hook that validates:
- Commit message analysis (checks for "optimal/perfect" claims)
- Python file changes validation
- Self-reflection checklist reminder
- CI status reminder
- Final gate before pushing to remote

### settings.local.json
Claude Code configuration with:
- Permission rules
- Pre-tool hooks that prompt reflection checks
- Auto-prompts before Write/Edit operations

## Installation

### 1. Install Git Hooks (One-Time Setup)

```bash
# Install pre-commit hook
cp .claude/pre-commit-hook .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit

# Install pre-push hook
cp .claude/pre-push-hook .git/hooks/pre-push
chmod +x .git/hooks/pre-push

# Or use the Makefile target (if available)
make install-hooks
```

### 2. Verify Installation

```bash
# Test pre-commit hook
git commit --allow-empty -m "Test pre-commit hook"

# Test pre-push hook (creates dummy commit and tries to push)
git commit --allow-empty -m "Test pre-push hook"
git push --dry-run # or just git push to your branch

# Should see: "✓ Self-reflection checks passed"
```

### 3. Skip Hooks (If Needed)

```bash
# Skip pre-commit
git commit --no-verify -m "Message"

# Skip pre-push
git push --no-verify

# Only skip when absolutely necessary
```

## Usage

### For Claude (AI Assistant)

The settings.local.json file automatically:
- Prompts reflection before Write/Edit operations
- Reminds to read SELF_REFLECTION.md
- Checks alignment with repo goals

### For Developers

1. **Before coding**: Read SELF_REFLECTION.md Quick Reference
2. **While coding**: Follow Code Patterns section
3. **Before committing**: Pre-commit hook validates changes
4. **Before pushing**: Pre-push hook validates commits + final checklist
5. **Review**: Check Pre-Change Reflection Checklist

## Quick Reference

### DO
- Use `config/{customer}/{platform}.yaml` for params
- Use `customer_paths.py` for data access
- Output actions as YAML with confidence + evidence
- Require high confidence before param updates
- Run all CI workflows before committing

### DON'T
- Hard-code paths or parameters
- Share params across customers/platforms
- Update params on low-confidence data
- Use synthetic data for evaluation claims
- Risk real money on unproven features

## Troubleshooting

### Hooks Not Running
```bash
# Check if executable
ls -la .git/hooks/pre-commit .git/hooks/pre-push

# Make executable
chmod +x .git/hooks/pre-commit
chmod +x .git/hooks/pre-push
```

### Pre-Commit Hook Blocking Valid Changes
1. Review the violation
2. Check SELF_REFLECTION.md
3. If truly valid: use `git commit --no-verify`
4. Consider updating the hook rules

### Pre-Push Hook Blocking Valid Changes
1. Review the commit messages for warning signs
2. Check SELF_REFLECTION.md
3. If truly valid: use `git push --no-verify`
4. Consider updating the hook rules

## Updating the Framework

To modify self-reflection rules:
1. Edit `.claude/SELF_REFLECTION.md`
2. Update `.claude/pre-commit-hook` checks
3. Update `.claude/pre-push-hook` checks
4. Update `.claude/settings.local.json` prompts
5. Copy hooks to `.git/hooks/`
6. Test with `git commit --allow-empty` and `git push --dry-run`

## Philosophy

This framework exists to ensure:
- **Reliability over theoretical optimization**
- **Real money protection** (no risky experiments)
- **Practical improvement** (beat history, not perfection)
- **Config-driven** (no hard-coding)
- **Multi-customer/platform support** (isolated configs)

The goal is NOT perfect budget allocation.
The goal IS solutions that beat or match historical performance.
