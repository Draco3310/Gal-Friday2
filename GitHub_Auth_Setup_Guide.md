# GitHub Authentication Setup Guide

## Best Practices for GitHub Authentication (2024)

### Option 1: GitHub CLI (Recommended for most users)
The easiest and most secure method for individual developers.

```bash
# Install GitHub CLI
# Windows (using Chocolatey or Scoop):
choco install gh
# or
scoop install gh

# Authenticate
gh auth login

# Select:
# - GitHub.com
# - HTTPS
# - Login with web browser (recommended)
# - Follow the prompts
```

**Pros:**
- Easy setup
- Handles token rotation automatically
- Works with 2FA
- Can manage multiple accounts

### Option 2: Personal Access Token (PAT) - Fine-grained
Best for CI/CD or when you need specific permissions.

1. Go to GitHub → Settings → Developer settings → Personal access tokens → Fine-grained tokens
2. Click "Generate new token"
3. Set expiration (recommend 90 days max)
4. Select repository access (specific repos recommended)
5. Set permissions:
   - **Contents**: Read/Write
   - **Pull requests**: Write
   - **Metadata**: Read (automatically selected)
   
6. Store the token securely and use it:
```bash
# Windows Git Bash or WSL
git config --global credential.helper manager
# When prompted for password, use the token instead
```

### Option 3: SSH Keys (Best for advanced users)
Most secure for long-term use.

```bash
# Generate SSH key
ssh-keygen -t ed25519 -C "your-email@example.com"

# Start ssh-agent
eval "$(ssh-agent -s)"

# Add key to agent
ssh-add ~/.ssh/id_ed25519

# Copy public key
cat ~/.ssh/id_ed25519.pub
# Add this to GitHub → Settings → SSH and GPG keys

# Configure git to use SSH
git remote set-url origin git@github.com:Draco3310/Gal-Friday2.git
```

### Option 4: Git Credential Manager (Windows)
Integrates with Windows Credential Store.

```bash
# Should be included with Git for Windows
git config --global credential.helper manager-core

# On first push, it will open a browser for authentication
```

## Security Best Practices

1. **Never commit credentials**
   - Use environment variables
   - Use .env files (always in .gitignore)
   - Use secret management tools

2. **Token/Key Rotation**
   - Rotate PATs every 90 days
   - Use expiring tokens when possible
   - Revoke unused tokens immediately

3. **Least Privilege**
   - Only grant necessary permissions
   - Use fine-grained PATs over classic
   - Limit repository access

4. **2FA is mandatory**
   - Enable two-factor authentication
   - Use authenticator apps over SMS

## For Gal-Friday2 Project

### Recommended Setup:
1. **Development**: Use GitHub CLI or SSH keys
2. **Production/Server**: Use fine-grained PAT with minimal permissions
3. **CI/CD**: Use GitHub Actions secrets or deployment keys

### Environment Variables for Trading Bot:
Create a `.env` file (never commit this!):
```bash
# API Keys (Examples - use your real keys)
KRAKEN_API_KEY=your_kraken_api_key
KRAKEN_API_SECRET=your_kraken_api_secret

# Database
DATABASE_URL=postgresql://user:pass@localhost/galfriday
INFLUXDB_TOKEN=your_influxdb_token

# GitHub (if needed for automated operations)
GITHUB_TOKEN=ghp_your_fine_grained_token

# Other services
REDIS_URL=redis://localhost:6379
LOG_LEVEL=INFO
```

### Quick Test Your Setup:
```bash
# After setting up authentication
git push -u origin fix/era001-safe-removals

# Or with gh CLI
gh auth status
gh pr create --title "Your PR title" --body "Your PR description"
```