import argparse

import os
import subprocess
import re


def parse_arguments():
    parser = argparse.ArgumentParser(description="Release script for dsmc_solver.")
    parser.add_argument("id", type=int, help="The id of testcase to run.")
    parser.add_argument(
        "--sample",
        action="store_true",
        required=False,
        help="Use CPU version of the program.",
    )

    parser.add_argument(
        "--no_build",
        action="store_true",
        required=False,
        help="Skip building the project.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        required=False,
        help="Build in debug mode.",
    )

    parser.add_argument(
        "--input_dir",
        type=str,
        default="assets/testcases",
        help="Directory containing input testcases.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory to store output results.",
    )
    parser.add_argument(
        "--save_log",
        action="store_true",
        required=False,
        help="Save the output log to a file.",
    )

    parser.add_argument(
        "--profile",
        action="store_true",
        required=False,
        help="Enable profiling mode.",
    )
    parser.add_argument(
        "--profile_visual",
        action="store_true",
        required=False,
        help="Output profile in visual format.",
    )
    parser.add_argument(
        "--profile_cpu",
        action="store_true",
        required=False,
        help="Enable CPU profiling mode.",
    )
    parser.add_argument(
        "--ncu",
        action="store_true",
        required=False,
        help="Check kernel occupancy.",
    )

    parser.add_argument(
        "--dry_run",
        action="store_true",
        required=False,
        help="Perform a dry run without executing the test case.",
    )
    parser.add_argument(
        "--no_srun",
        action="store_true",
        required=False,
        help="Do not use srun even for GPU execution.",
    )

    return parser.parse_args()


def run_build(args: argparse.Namespace):
    no_build = getattr(args, "no_build", False)
    sample = getattr(args, "sample", False)
    debug = getattr(args, "debug", False)

    args.build_type = "Debug" if debug else "Release"
    if no_build:
        return

    command = ["cmake", "--build", f"build/{args.build_type}", "-j"]
    if sample:
        command.extend(["--target", "sample"])
    subprocess.run(command, check=True)


def make_output_path(output_dir: str, filename: str):
    path = os.path.join(output_dir, filename)
    if os.path.exists(path):
        os.remove(path)
    return path


def run_testcase(args: argparse.Namespace) -> dict:
    id = getattr(args, "id", 0)
    sample = getattr(args, "sample", False)
    build_type = getattr(args, "build_type", "Release")

    input_dir = getattr(args, "input_dir", "assets/testcases")
    output_dir = getattr(args, "output_dir", "outputs")
    save_log = getattr(args, "save_log", False)

    profile = getattr(args, "profile", False)
    profile_visual = getattr(args, "profile_visual", False)
    profile_cpu = getattr(args, "profile_cpu", False)
    ncu = getattr(args, "ncu", False)

    dry_run = getattr(args, "dry_run", False)
    no_srun = getattr(args, "no_srun", False)

    result = {
        "input_path": None,
        "output_path": None,
        "valid_path": None,
        "log_path": None,
        "success": False,
        "elapsed_time": None,
    }

    executable_name = "sample" if sample else "dsmc_solver"
    executable_path = os.path.join("build", build_type, executable_name)
    input_path = os.path.join(input_dir, f"case-{id:02d}.yaml")

    if sample:
        output_dir = os.path.join(output_dir, "sample")

    os.makedirs(output_dir, exist_ok=True)
    output_path = make_output_path(output_dir, f"case-{id:02d}.out")
    valid_path = make_output_path(output_dir, f"case-{id:02d}.valid")
    log_path = make_output_path(output_dir, f"case-{id:02d}.log")

    ncu_path = make_output_path(output_dir, f"case-{id:02d}.ncu-rep")

    prof_path = make_output_path(output_dir, f"case-{id:02d}.prof")
    prof_visual_path = make_output_path(output_dir, f"case-{id:02d}.nvvp")
    prof_json_path = make_output_path(output_dir, f"case-{id:02d}.prof.json")

    command = [
        executable_path,
        input_path,
        output_path,
    ]
    if ncu:
        ncu_command = [
            "ncu",
            "--set",
            "full",
            "--target-processes",
            "all",
            "--log-file",
            ncu_path,
        ]
        command = ncu_command + command
    if profile and not ncu:
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        profile_command = ["nvprof"]
        if profile_cpu:
            profile_command.extend(["--cpu-profiling", "on"])
        if profile_visual:
            profile_command.extend(
                ["--output-profile", prof_visual_path, "--profile-api-trace", "all"]
            )
        else:
            profile_command.extend(["--log-file", prof_path])
        command = profile_command + command
    if not no_srun:
        srun_command = (
            "srun -A ACD114118 -N 1 -n 1 --gpus-per-node 1 --time=00:03:00".split(" ")
        )
        command = srun_command + command

    if dry_run:
        print("Dry run command:")
        print(" ".join(command))
        return result

    log = ""
    if save_log:
        with open(log_path, "w") as log_file:
            proc_result = subprocess.run(
                command, stdout=log_file, stderr=subprocess.STDOUT
            )
        with open(log_path, "r") as log_file:
            log = log_file.read()
    else:
        proc_result = subprocess.run(command, check=True)
        log = proc_result.stdout.decode("utf-8") if proc_result.stdout else ""

    if proc_result.returncode != 0:
        return result

    result["input_path"] = input_path
    result["output_path"] = output_path
    result["valid_path"] = valid_path
    if save_log:
        result["log_path"] = log_path

    if os.path.exists(output_path):
        result["success"] = True

    if os.path.exists(prof_path) and profile_visual:
        with open(prof_json_path, "w") as convert_output:
            subprocess.run(
                ["python", "nvprof2json/nvprof2json.py", prof_visual_path],
                check=True,
                stdout=convert_output,
            )

    elapsed_log = re.search(r"Elapsed: *([0-9.]+) us", log)
    if elapsed_log:
        result["elapsed_time"] = int(elapsed_log.group(1))

    return result


# def validate(input_path: str, output_path: str, valid_path: str) -> bool:
#     command = [
#         "assets/validation",
#         input_path,
#         output_path,
#     ]
#     with open(valid_path, "w") as valid_file:
#         proc_result = subprocess.run(
#             command, stdout=valid_file, stderr=subprocess.STDOUT
#         )
#     if proc_result.returncode != 0:
#         return False

#     return True


if __name__ == "__main__":
    args = parse_arguments()

    run_build(args)
    result = run_testcase(args)

    if result["elapsed_time"] is not None:
        print(f"Elapsed time: {result['elapsed_time']} us")

    # if result["success"]:
    #     is_valid = validate(
    #         result["input_path"], result["output_path"], result["valid_path"]
    #     )
    #     print(f"Validation result: {'Yes' if is_valid else 'No'}")
