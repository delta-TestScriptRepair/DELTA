# run.py

import time
import os
from code_process import run_full_pipeline

def count_h5_files(directory):
    if not os.path.exists(directory):
        return 0
    return len([f for f in os.listdir(directory) if f.endswith(".h5")])

if __name__ == "__main__":
    start_time = time.time()

    api_key =
    run_full_pipeline(api_key)

    end_time = time.time()
    duration = end_time - start_time

    # Count .h5 files in each directory
    failure_count = count_h5_files("failure_files")
    gpt_input_count = count_h5_files("gpt_input")
    output_count = count_h5_files("output_files")

    total_failed = failure_count + gpt_input_count
    total = total_failed + output_count
    success_ratio = (output_count / total) * 100 if total > 0 else 0

    print("\n✅ All processing completed")
    print(f"⏱ Total time: {duration:.2f} seconds")
    print(f"📊 Number of failed repairs: {total_failed}")
    print(f"📈 Number of successful repairs: {output_count}")
    print(f"🎯 Success rate: {success_ratio:.2f}%")
