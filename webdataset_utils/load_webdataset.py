"""
WebDataset Loader and Utilities

This module provides functions for loading and processing WebDataset shards 
using PyTorch DataLoader. The main functions include splitting sample keys 
into meaningful parts, decoding byte strings to UTF-8, and loading the dataset 
into a DataLoader.

Functions:
    custom_splitter(sample):
        Splits the sample key into parts and assigns the corresponding content 
        to the appropriate field.

    custom_decoder(sample):
        Decodes byte strings in the sample to UTF-8 strings.

    load_webdataset(shard_files, batch_size):
        Loads a WebDataset from the specified shard files and returns a DataLoader.

Usage:
    To use this module, first define the shard file pattern and batch size. Then, 
    call `load_webdataset` to get a DataLoader for the dataset.

    Example:
        shard_pattern = "output_dir/shard-{:06d}.tar"
        batch_size = 16
        dataloader = load_webdataset(shard_pattern, batch_size)

        for batch in dataloader:
            print(batch["sequence"], batch["target"], batch["set"], batch["validation"])

Author:
    [Your Name]
    [Your Email]

"""
import webdataset as wds
import torch

def custom_splitter(sample):
    """
    Splits the sample key into parts and assigns the corresponding content to the appropriate field.

    Args:
        sample (dict): A dictionary containing the sample data.

    Returns:
        dict or None: A dictionary with the fields 'sequence', 'target', 'set', 'validation', '__key__', and '__url__' 
                      if all parts are present, otherwise None.
    """
    key = sample["__key__"]
    parts = key.split("/")
    if len(parts) != 2:
        print(f"Sample {sample['__key__']} has an unexpected number of parts: {len(parts)}")
        return None
    
    sample_key, part_key = parts
    part_content = sample["txt"]
    print(f"Sample {sample_key}, Part {part_key}: {part_content}")

    # Assign the content to the appropriate field based on the part key
    if "sequence" in part_key:
        sample["sequence"] = part_content
    elif "target" in part_key:
        sample["target"] = part_content
    elif "set" in part_key:
        sample["set"] = part_content
    elif "validation" in part_key:
        sample["validation"] = part_content

    # Return the complete sample if all required fields are present
    if "sequence" in sample and "target" in sample and "set" in sample and "validation" in sample:
        return {
            "sequence": sample["sequence"],
            "target": sample["target"],
            "set": sample["set"],
            "validation": sample["validation"],
            "__key__": sample["__key__"],
            "__url__": sample["__url__"]
        }
    return None

def custom_decoder(sample):
    """
    Decodes byte strings in the sample to UTF-8 strings.

    Args:
        sample (dict): A dictionary containing the sample data.

    Returns:
        dict: A dictionary with all byte string values decoded to UTF-8 strings.
    """
    return {k: v.decode('utf-8') if isinstance(v, bytes) else v for k, v in sample.items()}

def load_webdataset(shard_files, batch_size):
    """
    Loads a WebDataset from the specified shard files and returns a DataLoader.

    Args:
        shard_files (str): The pattern for the shard files to load.
        batch_size (int): The number of samples per batch.

    Returns:
        DataLoader: A PyTorch DataLoader for the dataset.
    """
    dataset = (
        wds.WebDataset(shard_files)
        .map(custom_splitter)
        .map(custom_decoder)
        .to_tuple("sequence", "target", "set", "validation")
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=4)
    return dataloader
