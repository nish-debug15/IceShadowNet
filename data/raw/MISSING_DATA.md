# MISSING DATA

Real PDS4 DFSAR and OHRC tiles have not been downloaded yet.

**To get the real data:**
1. Register and log in to the ISRO PRADAN portal.
2. Open the Map Browse / Data Search tool (chmapbrowse).
3. Search for DFSAR data (instrument id: sar) and OHRC data (instrument id: ohr).
4. Locate the target crater region (e.g. Haworth, Shoemaker, Faustini) based on coordinates.
5. Download the .IMG and .XML files into data/raw/.
6. Run python scripts/verify_real_tile.py data/raw/<filename>.IMG to verify they can be read.

**Note:** Until these files are present, the ingest scripts run in synthetic fallback mode unless the --real flag is provided.
