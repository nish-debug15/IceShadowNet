# Missing PRADAN Data

The PDS4 DFSAR and OHRC tiles have not yet been downloaded to `data/raw/`.

**What is missing:**
- DFSAR dual-pol (HH/HV) SAR tiles (.IMG/.XML format)
- OHRC high-resolution optical imagery corresponding to the same lunar craters.

**Where to get it:**
You must manually download the tiles from the ISRO PRADAN portal as outlined in the project README.md:
- https://pradan.issdc.gov.in/pradan/

Once downloaded, place the `.IMG` and `.XML` files directly into `data/raw/` and re-run the pipeline.
