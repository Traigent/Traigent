#!/usr/bin/env python3
"""
Script to view TraiGent storage information and optimization results.
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json

from traigent.config.types import TraigentConfig
from traigent.storage.local_storage import LocalStorageManager


def main():
    print("🔍 TraiGent Storage Information")
    print("=" * 50)

    # Get storage path
    config = TraigentConfig()
    storage_path = config.get_local_storage_path()

    print(f"📁 Storage Path: {storage_path}")

    if not Path(storage_path).exists():
        print("❌ Storage directory doesn't exist yet")
        print("   Run an optimization to create it!")
        return

    # Initialize storage manager
    storage = LocalStorageManager(storage_path)

    # Get storage info
    info = storage.get_storage_info()
    print("\n📊 Storage Statistics:")
    print(f"   • Total Sessions: {info['total_sessions']}")
    print(f"   • Total Trials: {info['total_trials']}")
    print(f"   • Storage Size: {info['storage_size_mb']} MB")

    # List sessions
    sessions = storage.list_sessions()
    if sessions:
        print("\n📋 Recent Sessions:")
        for i, session in enumerate(sessions[:5]):  # Show last 5
            print(f"   {i+1}. {session.function_name} ({session.status})")
            print(f"      • Session ID: {session.session_id}")
            print(f"      • Trials: {session.completed_trials}")
            if session.best_score:
                print(f"      • Best Score: {session.best_score:.4f}")
            print(f"      • Created: {session.created_at}")
            print()
    else:
        print("\n📋 No optimization sessions found")

    # Show cache info
    cache_dir = Path(storage_path) / "cache"
    if cache_dir.exists():
        cache_files = list(cache_dir.rglob("*"))
        cache_size = sum(f.stat().st_size for f in cache_files if f.is_file())
        print("💾 Cache Information:")
        print(f"   • Cache Files: {len([f for f in cache_files if f.is_file()])}")
        print(f"   • Cache Size: {cache_size / 1024:.1f} KB")

    # Show usage tracking
    usage_file = Path(storage_path) / "usage.json"
    if usage_file.exists():
        with open(usage_file) as f:
            usage_data = json.load(f)

        records = usage_data.get("usage_records", [])
        if records:
            total_trials = sum(r.get("trials_count", 0) for r in records)
            total_cost = sum(r.get("cost_credits", 0) for r in records)

            print("\n💰 Usage Summary:")
            print(f"   • Total Optimization Runs: {len(records)}")
            print(f"   • Total Trials: {total_trials}")
            print(f"   • Total Cost: {total_cost:.4f} credits")
            print(f"   • Latest Run: {records[-1]['timestamp'][:19]}")

    print("\n🗂️ Directory Structure:")
    for root, _dirs, files in os.walk(storage_path):
        level = root.replace(storage_path, "").count(os.sep)
        indent = " " * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        sub_indent = " " * 2 * (level + 1)
        for file in files:
            print(f"{sub_indent}{file}")


if __name__ == "__main__":
    main()
