import pandas as pd
import os
import tarfile
import io

def csv_to_webdataset(csv_path, output_dir, shard_size=1000):
    df = pd.read_csv(csv_path)
    os.makedirs(output_dir, exist_ok=True)
    shard_id = 0

    for start in range(0, len(df), shard_size):
        shard_df = df[start:start + shard_size]
        shard_file = os.path.join(output_dir, f"shard-{shard_id:06d}.tar")
        with tarfile.open(shard_file, "w") as tar:
            for idx, row in shard_df.iterrows():
                sample_key = f"sample-{shard_id:06d}-{idx:06d}"
                sample = {
                    f"{sample_key}/sequence.txt": row['sequence'],
                    f"{sample_key}/target.txt": str(row['target']),
                    f"{sample_key}/set.txt": str(row['set']),
                    f"{sample_key}/validation.txt": str(row['validation'])
                }
                for key, value in sample.items():
                    tarinfo = tarfile.TarInfo(name=key)
                    tarinfo.size = len(value)
                    tar.addfile(tarinfo, io.BytesIO(value.encode("utf-8")))
        shard_id += 1

    print(f"Created webdataset shards in {output_dir}")
