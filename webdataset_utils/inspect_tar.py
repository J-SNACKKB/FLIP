import tarfile

def inspect_tar(tar_file):
    with tarfile.open(tar_file, "r") as tar:
        for member in tar.getmembers():
            print("Member:", member.name)
            f = tar.extractfile(member)
            content = f.read() if f else None
            print("Content:", content)
