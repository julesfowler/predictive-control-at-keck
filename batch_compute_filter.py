import threading
import subprocess


def run_script(alpha, order, frames, name):
    subprocess.run(["python", "k2ao_compute_filter.py", alpha, order, frames, name])

if __name__ == "__main__":
    script1_thread = threading.Thread(target=run_script, args=("1", "10", "30000", "testfilter.npy"))
    script2_thread = threading.Thread(target=run_script, args=("2", "20", "60000", "testfilter2.npy"))

    script1_thread.start()
    script2_thread.start()

    script1_thread.join()
    script2_thread.join()

    print("Both scripts have finished executing.")
