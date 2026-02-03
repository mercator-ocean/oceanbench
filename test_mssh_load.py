# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2


# test_mssh_loading.py
import xarray as xr
import os
import requests

MSSH_URL = "https://minio.dive.edito.eu/project-ml-compression/public/glorys12_mssh_2024.nc"
cache_path = "/tmp/glorys12_mssh_2024.nc"

print("🔍 Test 1: Direct download with requests")
try:
    response = requests.get(MSSH_URL, timeout=30)
    print(f"   Status: {response.status_code}")
    print(f"   Size: {len(response.content) / 1e6:.2f} MB")

    with open(cache_path, "wb") as f:
        f.write(response.content)
    print("   ✓ Downloaded successfully")
except Exception as e:
    print(f"   ❌ Error: {e}")

print("\n🔍 Test 2: Open with xarray")
try:
    if os.path.exists(cache_path):
        mssh_ds = xr.open_dataset(cache_path)
        print(" Opened successfully")
        print(f"   Shape: {mssh_ds['mssh'].shape}")
        print(f"   Variables: {list(mssh_ds.data_vars)}")
        print(f"   Coords: {list(mssh_ds.coords)}")
    else:
        print("   ❌ Cache file not found")
except Exception as e:
    print(f"   ❌ Error: {e}")

print("\n🔍 Test 3: Test interpolation")
try:
    mssh = mssh_ds["mssh"]
    # Test sur 3 points
    test_interp = mssh.interp(latitude=[45.0, 50.0, 55.0], longitude=[0.0, 5.0, 10.0], method="linear")
    print(f"   ✓ Interpolation works: {test_interp.values}")
except Exception as e:
    print(f"   ❌ Error: {e}")
