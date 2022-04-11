import requests, zipfile
import pandas as pd

def download_file(url):
  local_filename = url.split('/')[-1]
  # NOTE the stream=True parameter below
  with requests.get(url, stream=True) as r:
    r.raise_for_status()
    with open(local_filename, 'wb') as f:
      for chunk in r.iter_content(chunk_size=8192): 
        # If you have chunk encoded response uncomment if
        # and set chunk_size parameter to None.
        #if chunk: 
        f.write(chunk)
  return local_filename

def download_and_extract(url):
  fnm = download_file(url)
  with zipfile.ZipFile("SimLex-999.zip", 'r') as zip_ref:
    zip_ref.extractall(".")
    local_fn = zip_ref.filelist[2].filename
  return local_fn

def download_df(url):
  local_fn = download_and_extract(url)
  df = pd.read_csv(local_fn, sep="\t")
  return df