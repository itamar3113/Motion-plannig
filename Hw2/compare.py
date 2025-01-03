import os
import filecmp

dir1 = r"C:\Users\itama\Downloads\compare\Hw2Code (1)"
dir2 = r"C:\Users\itama\Downloads\compare\Hw2Code_-_updated (1)"

def compare_directories(dir1, dir2):
    # Create a dictionary to hold the files from both directories
    files1 = {}
    files2 = {}

    # Traverse through the first directory
    for root, dirs, files in os.walk(dir1):
        for file in files:
            relative_path = os.path.relpath(os.path.join(root, file), dir1)
            files1[relative_path] = os.path.join(root, file)

    # Traverse through the second directory
    for root, dirs, files in os.walk(dir2):
        for file in files:
            relative_path = os.path.relpath(os.path.join(root, file), dir2)
            files2[relative_path] = os.path.join(root, file)

    # Find common files, and files only in one of the directories
    common_files = set(files1.keys()).intersection(files2.keys())
    only_in_dir1 = set(files1.keys()) - set(files2.keys())
    only_in_dir2 = set(files2.keys()) - set(files1.keys())

    print("Files only in Directory 1:", only_in_dir1)
    print("Files only in Directory 2:", only_in_dir2)

    # Compare common files
    for file in common_files:
        file1 = files1[file]
        file2 = files2[file]
        if not filecmp.cmp(file1, file2, shallow=False):
            print(f"Files differ: {file}")
        else:
            print(f"Files are identical: {file}")

# Run the comparison
compare_directories(dir1, dir2)
