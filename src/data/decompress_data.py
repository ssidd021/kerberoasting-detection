import requests
from zipfile import ZipFile
from io import BytesIO
from pandas.io import json
import os.path as path
from tabulate import tabulate


def process_datasets(path, url):
    zipFileRequest = requests.get(url)
    zipFile = ZipFile(BytesIO(zipFileRequest.content))
    datasetJSONPath = zipFile.extract(zipFile.namelist()[0])

    df = json.read_json(path_or_buf=datasetJSONPath, lines=True)
    df.groupby(['Channel']).size().sort_values(ascending=False)

    # Dump dataframe to text file
    with open(path, "w+", encoding="utf-8") as f:
        f.write(tabulate(df, headers='keys'))


if __name__ == "__main__":
    data_dir = path.abspath(path.join(__file__, "../../..")) + "\\data\\processed\\"

    dc_sync_path = data_dir + "dc_sync.txt"
    dc_sync_url = 'https://raw.githubusercontent.com/OTRF/Security-Datasets/master/datasets/atomic/windows/credential_access/host/covenant_dcsync_dcerpc_drsuapi_DsGetNCChanges.zip'

    brute_force_path = data_dir + "brute_force.txt"
    brute_force_url = "https://raw.githubusercontent.com/OTRF/Security-Datasets/master/datasets/atomic/windows/lateral_movement/host/purplesharp_ad_playbook_I.zip"

    remote_process_injection = data_dir + "remote_process_injection.txt"
    remote_process_injection_url = "https://raw.githubusercontent.com/OTRF/Security-Datasets/master/datasets/atomic/windows/discovery/host/empire_getsession_dcerpc_smb_srvsvc_NetSessEnum.zip"

    apt_sim = data_dir + "apt_sim.txt"
    apt_sim_url = "https://raw.githubusercontent.com/OTRF/Security-Datasets/master/datasets/atomic/windows/other/aptsimulator_cobaltstrike.zip"

    process_datasets(brute_force_path, dc_sync_url)
    process_datasets(dc_sync_path, brute_force_url)
    process_datasets(remote_process_injection, remote_process_injection_url)
    process_datasets(apt_sim, apt_sim_url)
