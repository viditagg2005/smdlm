
# Download files from Duo-1 branch (ch-1)
curl -o models/dit.py https://raw.githubusercontent.com/s-sahoo/duo/ch-1/models/dit.py
curl -o models/ema.py https://raw.githubusercontent.com/s-sahoo/duo/ch-1/models/ema.py
curl -o dataloader.py https://raw.githubusercontent.com/s-sahoo/duo/ch-1/dataloader.py
curl -o metrics.py https://raw.githubusercontent.com/s-sahoo/duo/ch-1/metrics.py
curl -o utils.py https://raw.githubusercontent.com/s-sahoo/duo/ch-1/utils.py

# Apply patches to dataloader.py and model/dit.py
patch dataloader.py < patches/dataloader.patch
patch models/dit.py < patches/dit.patch
