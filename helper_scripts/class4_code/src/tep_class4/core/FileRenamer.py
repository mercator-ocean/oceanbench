import re
import sys
from tep_class4.core.utils import yaml_loader
import os

"""
    Class to rename observation files to common output names
"""


class FileRenamer:
    def __init__(self, dest, name_sys, log, yaml_path="patterns.yaml"):
        self.log = log
        self.dest = dest
        self.name_sys = name_sys
        module_dir = os.path.dirname(__file__)
        yaml_path = os.path.join(module_dir, yaml_path)
        self.patterns = yaml_loader(yaml_path)

    def create_name(self, filename, data_type, cl_src):
        typedata = "_".join([data_type, cl_src])
        self.log.info(f"{typedata=}")

        if typedata in self.patterns:
            pattern_info = self.patterns[typedata]
            match = re.search(pattern_info["pattern"], filename)
            if match:
                # Extract matched groups and assign to variables based on 'groups' keys
                matched_groups = {key: value for key, value in zip(pattern_info["groups"], match.groups())}
                # Construct the new name using formatted string with matched groups
                new_name = pattern_info["name_format"].format(name_sys=self.name_sys, **matched_groups)
            else:
                self.log.error(f"Error in the matching for {typedata}")
                sys.exit(1)
        else:
            new_name = filename  # No pattern found, retain original name

        # Rename the file
        name_out = f"{self.dest}{new_name}"
        self.log.info(f"Renamed to: {name_out}")
        return name_out
