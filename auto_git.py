import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import subprocess
import os

# ===== YAHAN APNA GITHUB LINK DAALO =====
GITHUB_REPO_LINK = "https://github.com/Adnan-1234/LANGGRAPH.git"
# ========================================

REPO_PATH = os.path.dirname(os.path.abspath(__file__))

class AutoGitHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if not event.is_directory:
            print(f"🔄 Change detected: {os.path.basename(event.src_path)}")
            subprocess.run(["git", "add", "."], cwd=REPO_PATH)
            subprocess.run(["git", "commit", "-m", "Auto push"], cwd=REPO_PATH)
            subprocess.run(["git", "push"], cwd=REPO_PATH)
            print("✅ Pushed to GitHub\n")

# Pehle check karo git init hua ya nahi
if not os.path.exists(os.path.join(REPO_PATH, ".git")):
    print("🔧 First time setup - Initializing git...")
    subprocess.run(["git", "init"], cwd=REPO_PATH)
    subprocess.run(["git", "remote", "add", "origin", GITHUB_REPO_LINK], cwd=REPO_PATH)
    subprocess.run(["git", "branch", "-M", "main"], cwd=REPO_PATH)
    print("✅ Git initialized with your repo link")

# Pehli baar sab files ko push karo
print("📤 Pushing all existing files to GitHub...")
subprocess.run(["git", "add", "."], cwd=REPO_PATH)
subprocess.run(["git", "commit", "-m", "Initial commit - all files"], cwd=REPO_PATH)
subprocess.run(["git", "push", "-u", "origin", "main"], cwd=REPO_PATH)
print("✅ Initial push complete!\n")

# Ab watch karo
observer = Observer()
observer.schedule(AutoGitHandler(), REPO_PATH, recursive=True)
observer.start()
print(f"🚀 Watching: {REPO_PATH}")
print("📁 Ab jo bhi file banayoge ya change karoge, auto push hoga!\n")

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    observer.stop()
observer.join()