import os
from load_webdataset import load_webdataset
from inspect_tar import inspect_tar

def load_and_print_webdataset(shard_pattern):
    # Find all existing shard files matching the pattern
    shard_files = [shard_pattern.format(i) for i in range(1000) if os.path.exists(shard_pattern.format(i))]
    if not shard_files:
        print("No shards found.")
        return

    dataloader = load_webdataset(shard_files, batch_size=16)

    for batch in dataloader:
        print("Batch keys:", batch.keys())
        for i, key in enumerate(batch["__key__"]):
            try:
                sequence = batch["sequence"][i]
                target = batch["target"][i]
                set_ = batch["set"][i]
                validation = batch["validation"][i]
                
                print(f"Sample {i}:")
                print(f"  sequence: {sequence}")
                print(f"  target: {target}")
                print(f"  set: {set_}")
                print(f"  validation: {validation}")
            except IndexError:
                print(f"Sample {i} has an unexpected structure.")
        break  # Only printing the first batch

if __name__ == "__main__":
    output_dir = 'output_dir'
    shard_pattern = os.path.join(output_dir, "shard-{:06d}.tar")

    print("Inspecting tar files in output directory...")
    for tar_file in [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith(".tar")]:
        inspect_tar(tar_file)
    
    print("\nLoading and printing WebDataset samples...")
    load_and_print_webdataset(shard_pattern)
