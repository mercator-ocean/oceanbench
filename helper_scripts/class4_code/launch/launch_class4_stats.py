import subprocess
import sys

def main():
    # Check for required input parameters
    if len(sys.argv) != 5:
        print("Usage: script.py <CONFIG> <datestart> <dateend> <typemod>")
        sys.exit(1)

    config, datestart, dateend, typemod = sys.argv[1:5]
    options = "--fcst --zarr"
    loglevel = 7
    python_bin = "python"
    script_path = "../src/tep_class4/cli/class4_stats_all.py"
    command = [
        python_bin,
        script_path,
        f"--config={config}",
        f"--date1={datestart}",
        f"--date2={dateend}",
        f"--typemod={typemod}",
        f"--loglevel={loglevel}",
    ] + options.split()

    try:
        subprocess.run(command, check=True)
        print("Run submitted successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error submitting job: {e}")

if __name__ == "__main__":
    main()
