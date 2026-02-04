
## Git Hooks Setup

This repository uses git hooks to enforce workflow standards. After cloning, run:

```bash
git config core.hooksPath .githooks
```

This prevents direct pushes to `main` branch and enforces pull request workflow. See [.githooks/README.md](.githooks/README.md) for details.
