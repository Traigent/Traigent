# Local SonarQube Setup

Run SonarQube locally using Docker to avoid consuming SonarCloud tokens.

## Quick Start

```bash
# Start SonarQube (first time takes ~2-3 minutes to initialize)
cd scripts/sonarqube-local
docker compose up -d

# Wait for it to be ready
docker compose logs -f sonarqube  # Watch until you see "SonarQube is operational"

# Or check status
curl -s http://localhost:9000/api/system/status | python3 -c "import sys,json; print(json.load(sys.stdin)['status'])"
```

## First-Time Setup

1. **Access the UI**: http://localhost:9000
2. **Login**: admin / admin (you'll be prompted to change password)
3. **Create a project token**:
   - Go to: My Account → Security → Generate Tokens
   - Name: `traigent-local`
   - Type: Project Analysis Token
   - Copy the token

4. **Save the token** in `.env.local`:
   ```bash
   echo 'export SONAR_LOCAL_TOKEN="your-token-here"' >> ../../.env.local
   ```

## Running Analysis

```bash
# From project root
cd REDACTED_TRAIGENT_ROOT/Traigent

# Source token
source .env.local

# Run analysis with local config
$HOME/sonar-scanner/bin/sonar-scanner \
  -Dproject.settings=scripts/sonarqube-local/sonar-project-local.properties \
  -Dsonar.token=$SONAR_LOCAL_TOKEN
```

Or use the Makefile target:

```bash
make sonar-local
```

## Checking Results

After analysis completes:
- **Web UI**: http://localhost:9000/dashboard?id=traigent-local
- **API**:
  ```bash
  curl -u "$SONAR_LOCAL_TOKEN:" "http://localhost:9000/api/issues/search?componentKeys=traigent-local&statuses=OPEN"
  ```

## Management Commands

```bash
# Stop SonarQube (keeps data)
docker compose stop

# Start again
docker compose start

# View logs
docker compose logs -f

# Complete cleanup (removes all data)
docker compose down -v

# Check resource usage
docker stats traigent-sonarqube traigent-sonarqube-db
```

## Comparison: Local vs Cloud

| Feature | Local SonarQube | SonarCloud |
|---------|-----------------|------------|
| Cost | Free (self-hosted) | Token-limited |
| Setup | One-time Docker setup | Pre-configured |
| Analysis | Same rules | Same rules |
| History | Local only | Cloud-stored |
| CI Integration | Manual | Automatic |
| Recommended for | Development | PR checks |

## Troubleshooting

### SonarQube won't start
```bash
# Check if ports are in use
lsof -i :9000

# Check Docker logs
docker compose logs sonarqube

# Elasticsearch needs more memory - increase vm.max_map_count
sudo sysctl -w vm.max_map_count=262144
```

### Analysis fails with "Not authorized"
- Regenerate token in SonarQube UI
- Ensure token is for "Project Analysis" type

### Out of memory
```bash
# Add to docker-compose.yml under sonarqube service:
# deploy:
#   resources:
#     limits:
#       memory: 4G
```

## Resource Requirements

- **RAM**: ~2-4GB for SonarQube + ~512MB for PostgreSQL
- **Disk**: ~1GB for Docker images + analysis data
- **CPU**: Minimal when idle, spikes during analysis
