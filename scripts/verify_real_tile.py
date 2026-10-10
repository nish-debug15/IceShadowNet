import sys
import rasterio
import numpy as np

def verify_tile(path):
    print(f"Verifying tile: {path}")
    try:
        with rasterio.open(path) as src:
            print(f"  Shape      : {src.shape}")
            print(f"  Dtype      : {src.dtypes[0]}")
            print(f"  CRS        : {src.crs}")
            print(f"  Bands      : {src.count}")
            print(f"  NoData     : {src.nodatavals}")
            
            for i in range(1, src.count + 1):
                band = src.read(i)
                valid = band[band != src.nodata] if src.nodata is not None else band
                print(f"  Band {i}     : Min={valid.min():.3f}, Max={valid.max():.3f}, Mean={valid.mean():.3f}")
        print("Verification SUCCESS.")
        sys.exit(0)
    except Exception as e:
        print(f"FATAL: Failed to open or parse tile. Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python verify_real_tile.py <path_to_img_or_xml>")
        sys.exit(1)
    verify_tile(sys.argv[1])
