import subprocess
import datetime
import os
import sys

def run_git_command(command, cwd):
    """Runs a git command in the specified directory."""
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        print(f"Success: {' '.join(command)}")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running: {' '.join(command)}")
        print(e.stderr)
        return False

def get_changes_summary(cwd):
    """Generates a summary of staged changes."""
    try:
        # Get status of staged files
        result = subprocess.run(
            ['git', 'diff', '--cached', '--name-status'],
            cwd=cwd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        lines = result.stdout.strip().split('\n')
        if not lines:
            return None

        added_pdfs = 0
        modified_bib = False
        other_changes = 0

        for line in lines:
            parts = line.split(maxsplit=1)
            if len(parts) < 2:
                continue
            
            status, filename = parts
            
            if status.startswith('A') and filename.lower().endswith('.pdf'):
                added_pdfs += 1
            elif filename.lower().endswith('.bib'):
                modified_bib = True
            else:
                other_changes += 1

        summary_parts = []
        if added_pdfs > 0:
            summary_parts.append(f"Added {added_pdfs} paper{'s' if added_pdfs > 1 else ''}")
        if modified_bib:
            summary_parts.append("Updated references")
        if other_changes > 0:
            summary_parts.append(f"Updated {other_changes} other file{'s' if other_changes > 1 else ''}")
            
        return ", ".join(summary_parts) if summary_parts else "Updated content"

    except subprocess.CalledProcessError:
        return "Updated content"

def main():
    # Define the target directory (paper directory)
    # Assuming this script is in the project root
    project_root = os.path.dirname(os.path.abspath(__file__))
    target_dir = os.path.join(project_root, 'paper')

    if not os.path.isdir(target_dir):
        print(f"Error: Directory not found: {target_dir}")
        sys.exit(1)

    print(f"Updating repository at: {target_dir}")

    # Git commands
    # 1. Add all changes
    if not run_git_command(['git', 'add', '.'], cwd=target_dir):
        print("Failed to add changes.")
        sys.exit(1)

    # 2. Get Change Summary
    description = get_changes_summary(target_dir)

    # 3. Commit
    # Check if there are changes to commit first to avoid empty commit error
    status_result = subprocess.run(
        ['git', 'status', '--porcelain'],
        cwd=target_dir,
        stdout=subprocess.PIPE,
        text=True
    )
    
    if not status_result.stdout.strip():
        print("No changes to commit.")
    else:
        # Get current date
        today = datetime.date.today().isoformat()
        commit_message = f"Auto-update: {today} - {description}"
        
        if not run_git_command(['git', 'commit', '-m', commit_message], cwd=target_dir):
            print("Failed to commit changes.")
            sys.exit(1)

    # 3. Push
    # Using HEAD to push the current branch to origin
    if not run_git_command(['git', 'push', 'origin', 'HEAD'], cwd=target_dir):
        print("Failed to push to origin. Please ensure the remote 'origin' is configured.")
        sys.exit(1)

    print("Daily update completed successfully.")

if __name__ == "__main__":
    main()
