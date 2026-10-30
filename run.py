from __future__ import annotations
import sys
import subprocess
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def print_banner():
    print("""
============================================================
  🌳 FP-GROWTH ALGORITHM & SHOPPING BASKET ANALYZER 🌳
  Discover what items people frequently buy together!
============================================================
""")


def main():
    print_banner()
    base_dir = Path(__file__).resolve().parent

    print("Welcome! How would you like to explore?")
    print("  [1] 🌐 Launch Web Dashboard (Recommended - visual & easiest to use)")
    print("  [2] 🛒 Quick Demo (Supermarket Groceries in 5 seconds)")
    print("  [3] 📂 Analyze a CSV File (Upload or point to your transactions)")
    print("  [4] ✍️  Type in Custom Transactions (Step-by-step terminal prompt)")
    print("  [5] 🧪 Run Automated Tests (Verify everything works correctly)")
    print("  [6] ❌ Exit")
    print()

    choice = input("Select an option (1-6) [Default: 1]: ").strip()
    if not choice:
        choice = "1"

    if choice == "1":
        print("\n🚀 Starting Web Dashboard... Opening in your browser...")
        app_path = base_dir / "app.py"
        try:
            subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)], check=True)
        except KeyboardInterrupt:
            print("\nDashboard stopped. Goodbye!")

    elif choice == "2":
        print("\n🛒 Running Supermarket Grocery Demo...\n")
        cli_path = base_dir / "cli.py"
        subprocess.run([
            sys.executable, str(cli_path),
            "--dataset", "grocery",
            "--min-support", "0.3",
            "--min-confidence", "0.6"
        ])

    elif choice == "3":
        print("\n📂 Enter the path to your CSV file.")
        print("  (Tip: Press Enter to use the included sample 'data/supermarket_baskets.csv')")
        csv_input = input("CSV Path: ").strip()
        if not csv_input:
            csv_input = str(base_dir / "data" / "supermarket_baskets.csv")

        cli_path = base_dir / "cli.py"
        subprocess.run([
            sys.executable, str(cli_path),
            "--csv", csv_input,
            "--min-support", "0.25",
            "--min-confidence", "0.6"
        ])

    elif choice == "4":
        print("\n✍️ Entering Interactive Mode...")
        cli_path = base_dir / "cli.py"
        subprocess.run([
            sys.executable, str(cli_path),
            "--interactive",
            "--min-support", "0.3",
            "--min-confidence", "0.6"
        ])

    elif choice == "5":
        print("\n🧪 Running test suite with pytest...\n")
        subprocess.run([
            sys.executable, "-m", "pytest",
            f"--rootdir={base_dir}",
            f"-o", f"cache_dir={base_dir / '.pytest_cache'}",
            str(base_dir / "tests")
        ])

    elif choice == "6":
        print("Goodbye!")
        sys.exit(0)

    else:
        print("Invalid choice. Please run 'python run.py' again.")


if __name__ == "__main__":
    main()
