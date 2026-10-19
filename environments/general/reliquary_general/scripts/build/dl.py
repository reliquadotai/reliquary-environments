import os, sys
from huggingface_hub import hf_hub_download
tok=open(os.path.expanduser("~/.config/reliquary/hf-subnet.token")).read().strip()
repo, rev, *files = sys.argv[1:]
for f in files:
    p=hf_hub_download(repo,f,repo_type="dataset",revision=rev,token=tok,local_dir="raw/"+repo.replace("/","__"))
    print(p, os.path.getsize(p))
