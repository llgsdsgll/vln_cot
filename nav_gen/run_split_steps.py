from main import read_args
from split_task import gen_step_task, split_traj


def main():
    args = read_args()
    print(f"Running split stages with sim_gpu_device={args.sim_gpu_device}")
    print(f"Requested RAM device={args.ram_device}")
    split_traj(args)
    gen_step_task(args)


if __name__ == "__main__":
    main()
