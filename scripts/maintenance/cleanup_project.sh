#!/bin/bash
# cleanup_project.sh - Traigent Project Reorganization Script
# Created: 2024-10-14
# Purpose: Clean up build artifacts, reorganize scripts/reports, reduce repo size

set -e  # Exit on error
shopt -s nullglob  # Expand unmatched globs to empty list

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DRY_RUN=false
VERBOSE=false
SKIP_TRAIGENT=false
SKIP_DEEPEVAL=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --help)
            echo "Usage: $0 [--dry-run] [--verbose]"
            echo "  --dry-run  Show what would be done without making changes"
            echo "  --verbose  Show detailed output"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

log() {
    echo -e "${BLUE}[$(date +%H:%M:%S)]${NC} $1"
}

warn() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

error() {
    echo -e "${RED}❌ $1${NC}"
}

success() {
    echo -e "${GREEN}✅ $1${NC}"
}

# Dry run wrapper
execute() {
    if [ "$DRY_RUN" = true ]; then
        echo -e "${YELLOW}[DRY-RUN]${NC} $*"
    else
        if [ "$VERBOSE" = true ]; then
            echo "Executing: $*"
        fi
        "$@"
    fi
}

# Safe move with collision detection
safe_move() {
    local src="$1"
    local dest="$2"
    
    if [ ! -e "$src" ]; then
        warn "Source does not exist: $src"
        return 1
    fi
    
    if [ -e "$dest" ]; then
        warn "Destination already exists: $dest"
        read -p "Overwrite? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log "Skipping: $src"
            return 0
        fi
    fi
    
    execute mv "$src" "$dest"
}

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ] || [ ! -d "traigent" ]; then
    error "Must be run from Traigent project root"
    exit 1
fi

echo "════════════════════════════════════════════════════════════"
echo "🧹 Traigent Project Cleanup & Reorganization"
echo "════════════════════════════════════════════════════════════"
if [ "$DRY_RUN" = true ]; then
    warn "DRY-RUN MODE: No changes will be made"
fi
echo ""

# Calculate current size
INITIAL_SIZE=$(du -sh . 2>/dev/null | cut -f1)
log "Current project size: $INITIAL_SIZE"
echo ""

# ============================================================================
# PHASE 0: Pre-flight Checks
# ============================================================================
log "Phase 0: Pre-flight checks..."

# Check for active virtual environment
if [ -n "$VIRTUAL_ENV" ]; then
    warn "Virtual environment is active: $VIRTUAL_ENV"
    read -p "Continue? This will NOT delete the active venv. (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Verify .traigent contains only cache data
if [ -d ".traigent" ]; then
    TRAIGENT_FILES=$(find .traigent -type f ! -name "*.json" ! -name "*.log" ! -name "*.txt" 2>/dev/null | wc -l)
    if [ "$TRAIGENT_FILES" -gt 0 ]; then
        warn ".traigent contains $TRAIGENT_FILES non-cache files"
        read -p "Skip .traigent deletion and continue? (Y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Nn]$ ]]; then
            SKIP_TRAIGENT=true
            log "Will preserve .traigent directory"
        fi
    fi
fi

# Check .deepeval for evaluation artifacts
if [ -d ".deepeval" ]; then
    DEEPEVAL_SIZE=$(du -sh .deepeval 2>/dev/null | cut -f1)
    warn ".deepeval directory exists ($DEEPEVAL_SIZE) - may contain evaluation history"
    read -p "Delete .deepeval? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        SKIP_DEEPEVAL=true
        log "Will preserve .deepeval directory"
    fi
fi

success "Pre-flight checks passed"
echo ""

# ============================================================================
# PHASE 1: Remove Build Artifacts & Caches (~4.3GB)
# ============================================================================
log "Phase 1: Removing build artifacts & caches..."

REMOVED_SIZE=0

# Virtual environments (check not active)
for venv_dir in venv .venv test_venv; do
    if [ -d "$venv_dir" ]; then
        if [[ "$VIRTUAL_ENV" == *"$venv_dir"* ]]; then
            warn "Skipping active virtual environment: $venv_dir"
            continue
        fi
        SIZE=$(du -sh "$venv_dir" 2>/dev/null | cut -f1)
        log "  Removing $venv_dir ($SIZE)..."
        execute rm -rf "$venv_dir"
        ((REMOVED_SIZE++))
    fi
done

# Cache directories
for cache_dir in .mypy_cache .ruff_cache; do
    if [ -d "$cache_dir" ]; then
        SIZE=$(du -sh "$cache_dir" 2>/dev/null | cut -f1)
        log "  Removing $cache_dir ($SIZE)..."
        execute rm -rf "$cache_dir"
        ((REMOVED_SIZE++))
    fi
done

# .traigent (with safeguard)
if [ -d ".traigent" ] && [ "$SKIP_TRAIGENT" = false ]; then
    SIZE=$(du -sh ".traigent" 2>/dev/null | cut -f1)
    log "  Removing .traigent ($SIZE)..."
    execute rm -rf ".traigent"
    ((REMOVED_SIZE++))
elif [ "$SKIP_TRAIGENT" = true ]; then
    log "  Preserving .traigent (skipped by user)"
fi

# .deepeval (with safeguard)
if [ -d ".deepeval" ] && [ "$SKIP_DEEPEVAL" = false ]; then
    SIZE=$(du -sh ".deepeval" 2>/dev/null | cut -f1)
    log "  Removing .deepeval ($SIZE)..."
    execute rm -rf ".deepeval"
    ((REMOVED_SIZE++))
elif [ "$SKIP_DEEPEVAL" = true ]; then
    log "  Preserving .deepeval (skipped by user)"
fi

# Build artifacts
for artifact in .coverage test_concurrent.db debug_output.log; do
    if [ -f "$artifact" ]; then
        SIZE=$(du -sh "$artifact" 2>/dev/null | cut -f1)
        log "  Removing $artifact ($SIZE)..."
        execute rm -f "$artifact"
        ((REMOVED_SIZE++))
    fi
done

# Python cache files
log "  Removing __pycache__ directories..."
if [ "$DRY_RUN" = false ]; then
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true
else
    echo -e "${YELLOW}[DRY-RUN]${NC} Would remove __pycache__ directories and *.pyc files"
fi

success "Phase 1 complete: Removed $REMOVED_SIZE items"
echo ""

# ============================================================================
# PHASE 2: Create Organized Directory Structure
# ============================================================================
log "Phase 2: Creating organized directory structure..."

execute mkdir -p scripts/code_review/archived
execute mkdir -p scripts/maintenance/archived
execute mkdir -p docs/reports/code_reviews
execute mkdir -p docs/reports/project_analysis

success "Phase 2 complete: Directory structure created"
echo ""

# ============================================================================
# PHASE 3: Move Review Scripts (with collision detection)
# ============================================================================
log "Phase 3: Organizing review scripts..."

MOVED_COUNT=0
for script in batch_code_quality_review.py batch_performance_review.py \
              batch_security_review.py auto_review_utils.py \
              comprehensive_review_generator.py batch_review_utils.sh; do
    if [ -f "$script" ]; then
        log "  Moving $script..."
        safe_move "$script" "scripts/code_review/archived/$script" && ((MOVED_COUNT++))
    fi
done

success "Phase 3 complete: Moved $MOVED_COUNT review scripts"
echo ""

# ============================================================================
# PHASE 4: Move Fix Scripts (with collision detection)
# ============================================================================
log "Phase 4: Organizing maintenance scripts..."

MOVED_COUNT=0
for script in fix_*.py; do
    if [ -f "$script" ]; then
        log "  Moving $script..."
        safe_move "$script" "scripts/maintenance/archived/$script" && ((MOVED_COUNT++))
    fi
done

success "Phase 4 complete: Moved $MOVED_COUNT fix scripts"
echo ""

# ============================================================================
# PHASE 5: Move Reports (with collision detection and safety checks)
# ============================================================================
log "Phase 5: Consolidating reports..."

MOVED_COUNT=0

# Review reports (explicit list, exclude README.md and CHANGELOG.md)
declare -a review_reports=(
    "ALL_REVIEWS_COMPLETE_FINAL_REPORT.md"
    "CODE_QUALITY_REVIEW_SUMMARY.md"
    "PERFORMANCE_REVIEW_COMPLETION_REPORT.md"
    "REVIEW_COMPLETION_REPORT.md"
    "SECURITY_REVIEW_COMPLETION_REPORT.md"
    "SOUNDNESS_CORRECTNESS_COMPLETION_REPORT.md"
)

for report in "${review_reports[@]}"; do
    if [ -f "$report" ]; then
        log "  Moving $report..."
        safe_move "$report" "docs/reports/code_reviews/$report" && ((MOVED_COUNT++))
    fi
done

# Project analysis reports
declare -a analysis_reports=(
    "PRUNING_ANALYSIS.md"
    "REFACTORING_PROPOSAL.md"
    "REFACTORING_SUMMARY.md"
    "REFACTORING_VALIDATION.md"
)

for report in "${analysis_reports[@]}"; do
    if [ -f "$report" ]; then
        log "  Moving $report..."
        safe_move "$report" "docs/reports/project_analysis/$report" && ((MOVED_COUNT++))
    fi
done

success "Phase 5 complete: Moved $MOVED_COUNT reports"
echo ""

# ============================================================================
# PHASE 6: Move Feature Requests (with collision detection)
# ============================================================================
log "Phase 6: Organizing feature requests..."

MOVED_COUNT=0
for request in feature_request.md tvar_feature_request.md; do
    if [ -f "$request" ]; then
        log "  Moving $request..."
        safe_move "$request" "docs/feature_requests/$request" && ((MOVED_COUNT++))
    fi
done

success "Phase 6 complete: Moved $MOVED_COUNT feature requests"
echo ""

# ============================================================================
# PHASE 7: Remove Obsolete Items
# ============================================================================
log "Phase 7: Removing obsolete files..."

REMOVED_COUNT=0

if [ -d ".archive" ]; then
    log "  Removing .archive/ directory..."
    execute rm -rf .archive/
    ((REMOVED_COUNT++))
fi

if [ -L "demos_fundamentals" ]; then
    log "  Removing demos_fundamentals symlink..."
    execute rm -f demos_fundamentals
    ((REMOVED_COUNT++))
fi

if [ -f "usercustomize.py" ]; then
    log "  Removing usercustomize.py..."
    execute rm -f usercustomize.py
    ((REMOVED_COUNT++))
fi

success "Phase 7 complete: Removed $REMOVED_COUNT obsolete items"
echo ""

# ============================================================================
# PHASE 8: Git Repository Cleanup (SKIPPED - Active Work)
# ============================================================================
log "Phase 8: Git repository status..."

warn "Preserving active work directories:"
RECENT_FILES=$(find out/ -type f -mtime -7 2>/dev/null | wc -l)
log "  - out/ contains $RECENT_FILES recently modified files (active code reviews)"
log "  - reports/ contains ongoing documentation"
log "  - Not untracking from git to preserve active work"

success "Phase 8 complete: Active work directories preserved"
echo ""

# ============================================================================
# PHASE 9: Update .gitignore
# ============================================================================
log "Phase 9: Updating .gitignore..."

if [ "$DRY_RUN" = false ]; then
    # Check if patterns already exist
    NEEDS_UPDATE=false

    if ! grep -q "issues_viewer/node_modules/" .gitignore; then
        echo "" >> .gitignore
        echo "# Issues viewer build artifacts (added by cleanup script)" >> .gitignore
        echo "issues_viewer/node_modules/" >> .gitignore
        echo "issues_viewer/dist/" >> .gitignore
        NEEDS_UPDATE=true
    fi

    if [ "$NEEDS_UPDATE" = true ]; then
        success "Phase 9 complete: .gitignore updated"
    else
        success "Phase 9 complete: .gitignore already up to date"
    fi
else
    log "  [DRY-RUN] Would update .gitignore for issues_viewer/"
fi
echo ""

# ============================================================================
# Summary
# ============================================================================
echo "════════════════════════════════════════════════════════════"
echo "📊 Cleanup Summary"
echo "════════════════════════════════════════════════════════════"

FINAL_SIZE=$(du -sh . 2>/dev/null | cut -f1)
log "Initial size: $INITIAL_SIZE"
log "Final size:   $FINAL_SIZE"
echo ""

if [ "$DRY_RUN" = true ]; then
    warn "DRY-RUN MODE: No changes were made"
    echo ""
    echo "To execute cleanup, run:"
    echo "  bash scripts/maintenance/cleanup_project.sh"
else
    success "Cleanup complete!"
    echo ""
    echo "Next steps:"
    echo "  1. Review changes: git status"
    echo "  2. Test build:     make test"
    echo "  3. Commit:         git add -A && git commit -m 'refactor: reorganize project structure'"
fi

echo "════════════════════════════════════════════════════════════"
