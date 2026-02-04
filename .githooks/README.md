# Git Hooks

This repository uses git hooks to enforce code quality and workflow standards.

## Pre-push Hook

The `pre-push` hook prevents direct pushes to the `main` and `master` branches, enforcing a pull request workflow.

### Protection

- **Blocked branches**: `main`, `master`
- **Allowed branches**: All other branches (feature/*, bugfix/*, etc.)

### Enable Hooks (One-time Setup)

After cloning the repository, run:

```bash
git config core.hooksPath .githooks
```

This tells git to use hooks from the `.githooks` directory instead of `.git/hooks`.

### Verify Installation

```bash
ls -la .githooks/pre-push
git config --get core.hooksPath
```

Expected output:
```
-rwxr-xr-x  ... .githooks/pre-push
.githooks
```

### Workflow

**✅ Correct (feature branch):**
```bash
git checkout -b feature/my-feature
git commit -m "Add feature"
git push -u origin feature/my-feature  # ✓ Allowed
# Then create PR on GitHub
```

**❌ Incorrect (direct to main):**
```bash
git checkout main
git commit -m "Add feature"
git push origin main  # ✗ BLOCKED by hook
```

### Emergency Bypass (Not Recommended)

If you absolutely must push directly to main (e.g., fixing the hook itself):

```bash
git push --no-verify origin main
```

### Adding New Hooks

1. Create the hook file in `.githooks/`
2. Make it executable: `chmod +x .githooks/your-hook`
3. Commit it to the repository
4. Team members will automatically get it via `git pull`

### Hook Reference

| Hook | Purpose | Status |
|------|---------|--------|
| `pre-push` | Block direct pushes to main/master | ✅ Active |
