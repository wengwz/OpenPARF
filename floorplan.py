
import os
import sys
import argparse
from loguru import logger

from openparf.flow import island_place

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="Island Placement Framework")
    
    parser.add_argument('--netlist', type=str, required=True, help="path to the partition results JSON file")
    parser.add_argument('--config', type=str, required=True, help='path to the config JSON file')
    parser.add_argument('--log', type=str, default=None, help='path to the logging file')
    args = parser.parse_args()
    
    netlist_file = args.netlist
    config_file = args.config
    log_file = args.log
    
    # Set up logging
    log_format = "<level>[{level:<7}]</level> <green>{elapsed} sec</green> | {name}:{line} - {message}"
    logger.configure(handlers=[{"sink": sys.stderr, "format": log_format, "level": "INFO"}])
    if log_file is not None:
        logger.add(log_file, colorize=True, format=log_format, level="INFO")
    
    logger.info(f"Command Argument List: {args}")
    
    island_place(config_file, netlist_file)