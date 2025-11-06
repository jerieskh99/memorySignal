import os
import subprocess


def iterate_delete_non_hdf5(curr_path):
    if not os.path.isfile(curr_path):
        if curr_path.endswith('.csv') or curr_path.endswith('.json'):
            os.remove(curr_path)
        return
    
    if os.path.isdir(curr_path):
        sub_dirs = os.listdir(curr_path)

        if sub_dirs == []:
            return
    
        for d in sub_dirs:
            iterate_delete_non_hdf5(os.path.join(curr_path, d))


def main():
    base_dir = '/Volumes/Extreme SSD/thesis/runs/mixed'
    iterate_delete_non_hdf5(base_dir)


if __name__ == '__main__':
    main()
