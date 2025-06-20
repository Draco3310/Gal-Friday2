# Git Repository Cleanup Guide

## Current Issues Found

1. **.idea folder is being tracked** - This PyCharm/IDE folder should not be in version control
2. **Merge conflict in .gitignore** - Now fixed
3. **Missing important .gitignore entries** - Now added

## Steps to Clean Up

### 1. Remove .idea from tracking
```bash
# Remove .idea from git tracking but keep it locally
git rm -r --cached .idea/

# Commit the removal
git add .gitignore
git commit -m "Remove .idea folder from tracking and update .gitignore"
```

### 2. Check for other files that shouldn't be tracked
```bash
# List all tracked files
git ls-files > tracked_files.txt

# Review for sensitive files like:
# - Config files with credentials
# - .env files
# - Database files
# - API keys
# - Log files

# Check specifically for common sensitive patterns
git ls-files | grep -E "(\.env|config.*local|\.log$|\.db$|\.sqlite|credentials|secret)" 

# If you find any, remove them:
git rm --cached <filename>
```

### 3. Create example/template files
```bash
# Create templates for sensitive config files
cp config.yaml config.example.yaml
# Edit config.example.yaml to remove all sensitive data, leave only structure

# Create .env template
cat > .env.example << 'EOF'
# API Keys
KRAKEN_API_KEY=your_api_key_here
KRAKEN_API_SECRET=your_api_secret_here

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/galfriday
INFLUXDB_URL=http://localhost:8086
INFLUXDB_TOKEN=your_token_here

# Redis
REDIS_URL=redis://localhost:6379

# Logging
LOG_LEVEL=INFO

# Trading Configuration
PAPER_TRADING=true
MAX_POSITION_SIZE=1000
EOF

git add config.example.yaml .env.example
git commit -m "Add example configuration files"
```

### 4. Set up git hooks to prevent accidental commits
Create `.git/hooks/pre-commit`:
```bash
#!/bin/sh
# Prevent committing sensitive files

SENSITIVE_PATTERNS="\.env$|config.*local|credentials|secret|\.pem$|\.key$"

# Check if any sensitive files are being committed
SENSITIVE_FILES=$(git diff --cached --name-only | grep -E "$SENSITIVE_PATTERNS")

if [ ! -z "$SENSITIVE_FILES" ]; then
    echo "Error: Attempting to commit sensitive files:"
    echo "$SENSITIVE_FILES"
    echo "Please remove these files from the commit using: git reset HEAD <file>"
    exit 1
fi

exit 0
```

Make it executable:
```bash
chmod +x .git/hooks/pre-commit
```

## Recommended Project Structure

```
Gal-Friday2/
├── .env.example          # Template for environment variables
├── .gitignore           # Updated with comprehensive rules
├── config/
│   ├── config.example.yaml  # Template configuration
│   ├── feature_registry.yaml # Can be tracked (no secrets)
│   └── local.yaml       # Should NOT be tracked (in .gitignore)
├── gal_friday/          # Source code
├── tests/               # Test files
├── docs/                # Documentation
└── scripts/             # Utility scripts
```

## Best Practices Going Forward

1. **Never commit sensitive data**
   - Always use .env files for secrets
   - Use config templates with dummy values
   - Review changes before committing

2. **Use environment variables**
   ```python
   import os
   from dotenv import load_dotenv
   
   load_dotenv()
   
   API_KEY = os.getenv('KRAKEN_API_KEY')
   ```

3. **Regular audits**
   ```bash
   # Periodically check for large files
   git ls-files -s | sort -n -k 2 | tail -20
   
   # Check for potential secrets
   git grep -E "(password|secret|key|token)" -- "*.py" "*.yaml" "*.json"
   ```

4. **Use git-secrets or similar tools**
   ```bash
   # Install git-secrets
   # Prevents committing AWS keys and other patterns
   ```

## After Cleanup

Your repository will be cleaner and more secure. Team members will need to:
1. Create their own local config files from templates
2. Set up their own .env file
3. Configure their IDE settings