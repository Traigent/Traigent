import os
import re
from pathlib import Path

from traigent.utils.secure_path import PathTraversalError, safe_open, validate_path

def patch_examples(root_dir):
    base_dir = Path.cwd()
    try:
        root_dir = validate_path(root_dir, base_dir, must_exist=True)
    except (PathTraversalError, FileNotFoundError) as exc:
        raise SystemExit(f"Error: {exc}") from exc

    print(f"Patching examples in {root_dir} for mock mode...")
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".py"):
                filepath = validate_path(Path(root) / file, root_dir, must_exist=True)
                with safe_open(filepath, root_dir, mode="r", encoding="utf-8") as f:
                    lines = f.readlines()
                
                new_lines = []
                modified = False
                
                i = 0
                while i < len(lines):
                    line = lines[i]
                    new_lines.append(line)
                    
                    if line.strip().startswith("def ") and ("):" in line or ") ->" in line):
                        print(f"Found function definition: {line.strip()}")
                        # Check if decorated with @traigent.optimize
                        is_decorated = False
                        # Look backwards
                        for j in range(i-1, max(-1, i-20), -1):
                            prev_line = lines[j].strip()
                            print(f"Checking line {j}: {prev_line}")
                            if "@traigent.optimize" in prev_line:
                                is_decorated = True
                                break
                            if prev_line == "" or prev_line.startswith("#"):
                                continue
                            if not prev_line.startswith("@") and not prev_line.startswith(")"):
                                # Likely not part of the decorator stack we care about
                                break
                        
                        if is_decorated:
                            # Check if next lines already have mock check
                            has_mock_check = False
                            for k in range(i+1, min(len(lines), i+10)):
                                if "TRAIGENT_MOCK_LLM" in lines[k]:
                                    has_mock_check = True
                                    break
                            
                            if not has_mock_check:
                                # Determine indentation
                                indent = line[:line.find("def")] + "    "
                                
                                # Add mock check
                                new_lines.append(f'\n{indent}# Check for mock mode\n')
                                new_lines.append(f'{indent}import os\n')
                                new_lines.append(f'{indent}if os.environ.get("TRAIGENT_MOCK_LLM", "false").lower() == "true":\n')
                                new_lines.append(f'{indent}    return "Mock response"\n')
                                modified = True
                    
                    i += 1
                            
                if modified:
                    print(f"Patched {filepath}")
                    with safe_open(
                        filepath, root_dir, mode="w", encoding="utf-8"
                    ) as f:
                        f.writelines(new_lines)

if __name__ == "__main__":
    patch_examples(Path("examples/docs/page-inline"))
