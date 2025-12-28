# 🔧 Maintenance Tools

This directory contains **active maintenance tools** for the Traigent project. These are safe, well-tested tools that should be used for ongoing project maintenance.

## 🛠️ **Available Tools**

### 📊 Code Quality Management
**`traigent_quality_manager.py`** - Comprehensive code quality tool
- Unified reporting (flake8, ruff, mypy, custom checks)
- Safe automated fixing with dry-run and rollback
- Automatic backups before making changes
- Interactive and automated modes

```bash
# Generate quality report
python traigent_quality_manager.py --report

# Preview fixes
python traigent_quality_manager.py --fix --dry-run

# Apply fixes interactively
python traigent_quality_manager.py --fix

# Apply fixes automatically
python traigent_quality_manager.py --fix --auto-yes

# List backups
python traigent_quality_manager.py --list-backups

# Rollback changes
python traigent_quality_manager.py --rollback <backup_id>
```

### 🧹 Project Cleanup
**`safe_cleanup_manager.py`** - Comprehensive project cleanup tool
- Safe removal of temporary files, caches, and old reports
- Automatic backup before cleanup
- Interactive and automated modes
- Undo capability

```bash
# Analyze cleanup opportunities
python safe_cleanup_manager.py --analyze

# Preview cleanup
python safe_cleanup_manager.py --cleanup --dry-run

# Interactive cleanup
python safe_cleanup_manager.py --cleanup --interactive

# Automated cleanup
python safe_cleanup_manager.py --cleanup --auto

# List cleanup backups
python safe_cleanup_manager.py --list-backups

# Undo cleanup
python safe_cleanup_manager.py --undo <backup_id>
```

### 📈 Performance Monitoring
**`performance_monitor.py`** - System performance monitoring
- Monitor resource usage during development
- Track performance metrics over time
- Generate performance reports

## ⚠️ **Safety Features**

All maintenance tools include:
- **Dry-run mode** - Preview changes without applying them
- **Automatic backups** - Create backups before making changes
- **Rollback capability** - Undo changes if needed
- **Interactive confirmation** - Ask before destructive operations
- **Comprehensive logging** - Track all operations
- **Syntax validation** - Verify code changes before applying

## 🔄 **Backup Management**

Both tools create backups in:
- **Quality backups**: `scripts/maintenance/backups/`
- **Cleanup backups**: `scripts/maintenance/cleanup_backups/`

Backup IDs are timestamped for easy identification.

## 📋 **Best Practices**

1. **Always run with `--dry-run` first** to preview changes
2. **Use interactive mode** for important operations
3. **Keep backups** until you've verified changes work correctly
4. **Check logs** in `scripts/maintenance/logs/` if issues occur
5. **Test after changes** by running the project tests

## 🚨 **Emergency Recovery**

If a maintenance operation goes wrong:

1. **Check recent backups**: `python <tool>.py --list-backups`
2. **Restore from backup**: `python <tool>.py --rollback <backup_id>`
3. **Check logs**: Look in `scripts/maintenance/logs/` for details
4. **Verify restoration**: Run tests to confirm everything works

## 📅 **Maintenance Schedule**

**Weekly:**
- Run quality analysis: `traigent_quality_manager.py --report`
- Clean up temporary files: `safe_cleanup_manager.py --cleanup --interactive`

**Monthly:**
- Full quality check and fix: `traigent_quality_manager.py --fix`
- Review and clean old reports and logs

**As Needed:**
- Performance monitoring during development
- Quality fixes before releases
- Cleanup before major refactoring

---
*Last updated: September 11, 2025*
