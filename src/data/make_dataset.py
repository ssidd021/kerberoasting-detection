# -*- coding: utf-8 -*-
import logging
import re
import xml.etree.ElementTree as et
from collections import defaultdict
from pathlib import Path
import pandas as pd
import os
from tabulate import tabulate
from dotenv import find_dotenv, load_dotenv


def process_xml_to_df(dir, attack):
    xtree = et.parse(dir)
    root = xtree.getroot()
    system_data = defaultdict(list)

    for events in root.iter('Events'):
        root1 = events
        for event1 in root1:  # event
            for root2 in event1:
                tag_name = re.sub(r"{.*.}", "", root2.tag)
                if tag_name == "System":
                    for system in root2.iter():
                        if system.attrib.get("Guid"):
                            system_data["provider_guid"].append(system.attrib.get("Guid"))

                        tag = re.sub(r"{.*.}", "", system.tag)
                        if tag != "System":
                            system_data[tag].append(system.text)
                elif tag_name == "EventData":
                    for event2 in root2.iter():
                        tag = re.sub(r"{.*.}", "", event2.tag)
                        if tag != "EventData":
                            system_data[event2.attrib.get("Name")].append(event2.text)

    df = pd.DataFrame.from_dict(system_data, orient='index').transpose()
    file_name = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'data\\processed')) + "\\" + attack + ".txt"

    with open(file_name, "w+") as f:
        f.write(tabulate(df, headers='keys'))


def main():
    """ Runs data processing scripts to turn raw data from (../processed) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('creating final dataset from raw data')

    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'data\\interim'))
    attacks = ["password_spraying", "dc_shadow", "kerberoasting", "golden_ticket"]

    for attack in attacks:
        path = data_dir + "\\" + attack + ".xml"
        process_xml_to_df(path, attack)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
