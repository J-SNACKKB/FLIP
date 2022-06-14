import os
import shutil
from pathlib import Path

import yaml

import subprocess
import logging

from utils import *

import argparse

split_dict = {
#    'aav_1': 'des_mut',
#    'aav_2': 'mut_des',
#    'aav_3': 'one_vs_many',
#    'aav_4': 'two_vs_many',
#    'aav_5': 'seven_vs_many',
#    'aav_6': 'low_vs_high',
#    'aav_7': 'sampled',
#    'meltome_1' : 'mixed_split',
#    'meltome_2' : 'human',
#    'meltome_3' : 'human_cell',
#    'gb1_1': 'one_vs_rest',
#    'gb1_2': 'two_vs_rest',
#    'gb1_3': 'three_vs_rest',
#    'gb1_4': 'sampled',
#    'gb1_5': 'low_vs_high',
    '2str': 'sampled',
    'conservations': 'sampled'
}

configs_bank = Path('') / 'configsbank'
splits = Path('') / '..' / 'splits'

def create_parser():
    parser = argparse.ArgumentParser(description="Train and evaluate different bioembedding models using biotrainer.")
    parser.add_argument("split", choices = split_dict.keys(), type=str, help="The split to train and evaluate. Options: {}.".format(split_dict.keys()))
    parser.add_argument("protocol", choices=['residue_to_class', 'sequence_to_class', 'sequence_to_value', 'residue_to_value'], type=str, help="The protocol to use. Options: residue_to_class, sequence_to_class, sequence_to_sequence.")
    parser.add_argument("-e", "--embedder", type=str, help="The embedder to use.")
    parser.add_argument("-m", "--model", type=str, help="The model to use.")
    parser.add_argument("-c", "--config", help="Config file different from the provided one in configsbank.", type=str, default=None)

    return parser

def prepare_data(split, protocol, working_dir):
    # Check if the sequence.fasta and labels.fasta files exists
    if os.path.exists(working_dir / 'sequence.fasta') and os.path.exists(working_dir / 'labels.fasta'):
        logging.info('Sequence and labels files already exists.')
        logging.info('Skipping data preparation.')
    else:
        # Conversion CSV to FASTA
        split_dir = splits / split.split('_')[0] / 'splits' / (split_dict[split] + '.csv')
        destination_sequences_dir = working_dir / 'sequences.fasta'
        destination_labels_dir = working_dir / 'labels.fasta'
        
        if protocol == 'residue_to_class':
            logging.info('Converting CSV to FASTA for residue to class protocol.')
            residue_to_class_fasta(split_dir, destination_sequences_dir, destination_labels_dir)
            return destination_sequences_dir, destination_labels_dir
        elif protocol == 'sequence_to_class':
            logging.info('Converting CSV to FASTA for sequence to class protocol.')
            # TODO: Standardization pending in biotrainer and FLIP
        elif protocol == 'sequence_to_value':
            logging.info('Converting CSV to FASTA for sequence to value protocol.')
            protein_to_class_fasta(split_dir, destination_sequences_dir)
            return destination_sequences_dir, None
        elif protocol == 'residue_to_value':
            logging.info('Converting CSV to FASTA for residue to value protocol.')
            protein_to_value_fasta(split_dir, destination_sequences_dir)
            return destination_sequences_dir, None

def main(args):
    # Get path of the configuration file from configsbank or the provided one
    if args.config is None:
        config_file = configs_bank / (args.split + '.yml')
    else:
        config_file = Path(args.config)
    logging.info('The selected configuration file is: {}'.format(config_file))

    # Create folder for temporary files and results
    if not os.path.exists('./results-{}'.format(args.split)):
        os.makedirs('./results-{}'.format(args.split))
    logging.info('Temporary files and results will be saved in ./results-{}'.format(args.split))
    
    # Set working_dir
    working_dir = Path('./results-{}'.format(args.split))

    # Create copy of the configuration file
    shutil.copyfile(config_file, './results-{}/config.yml'.format(args.split))

    # Prepare the data
    sequences, labels = prepare_data(args.split, args.protocol, working_dir)
    logging.info('Data prepared')

    # Modify config file with correct data of the selected split and the selected embedding
    with open('./results-{}/config.yml'.format(args.split), 'r') as cfile:
        config = yaml.load(cfile, Loader=yaml.FullLoader)

    for item in config:
        if item == "sequence_file":
            logging.info('Modifying config file with sequence file: {}'.format(sequences))
            config["sequence_file"] = str(sequences).split('/')[-1]
        elif item == "labels_file" and labels is not None:
            logging.info('Modifying config file with labels file: {}'.format(labels))
            config["labels_file"] = str(labels).split('/')[-1]
        elif item == "embedder_name" and args.embedder is not None:
            logging.info('Config file uses {} embedder. Changed by {}'.format(item["value"], args.embedder))
            config["embedder_name"] = args.embedder
        elif item == "model_choice" and args.model is not None:
            logging.info('Config file uses {} model. Changed by {}'.format(item["value"], args.model))
            config["model_choice"] = args.model

    with open('./results-{}/config.yml'.format(args.split), 'w') as cfile:
        yaml.dump(config, cfile)

    # Run biotrainer
    logging.info('Executing biotrainer.')
    #subprocess.call(["poetry", "run", "biotrainer", "./results-{}/config.yml".format(args.split)])
    os.chdir(working_dir)
    subprocess.call(["python3", "../../../biotrainer/run-biotrainer.py", "./config.yml"])
    logging.info('Done.')

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    
    parser = create_parser()
    args = parser.parse_args()
    main(args)
