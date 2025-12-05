.. SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
..
.. SPDX-License-Identifier: EUPL-1.2

.. _input-datasets-for-oceanbench-challenger-evaluation-page:

=============================================================
 Input datasets for OceanBench challenger evaluation
=============================================================


This page lists the initial states (GLO2 nowcasts) and forcing (IFS) datasets for OceanBench challengers to produce forecast datasets to evaluate in the benchmark.


You can open and explore these datasets by using the ``oceanbench.datasets.input`` module, the documentation is `here <https://oceanbench.readthedocs.io/en/latest/source/oceanbench.datasets.html#module-oceanbench.datasets.input>`_.

Dataset files
**********************************************

The dataset consists of two families of files:

* **NetCDF files** (``.nc``) – raw model output.
* **Zarr collections** (``.zarr``) – chunked, cloud‑optimised arrays.

Both families are listed below with their download URLs and MD5 checksums
(where available).

.. csv-table::
   :header: "File name","Download URL","Format","MD5 checksum"
   :widths: 20,45,15,20
   :align: left

   "IFS_20240221.nc", "https://minio.dive.edito.eu/project-oceanbench/public/IFS/IFS_20240221.nc", "application/x-netcdf", "e15f06b6a7621b95bc450cb34e838486"
   "IFS_20241030.nc", "https://minio.dive.edito.eu/project-oceanbench/public/IFS/IFS_20241030.nc", "application/x-netcdf", "c6a8eb4d3525369a6b832f6ab8c912ca"
   "IFS_20240814.nc", "https://minio.dive.edito.eu/project-oceanbench/public/IFS/IFS_20240814.nc", "application/x-netcdf", "23c89b6802a8068f9a463fb040f9996a"
   "IFS_20241113.nc", "https://minio.dive.edito.eu/project-oceanbench/public/IFS/IFS_20241113.nc", "application/x-netcdf", "2596b08d15724970d3a1ad47826a509e"
   "IFS_20240410.nc", "https://minio.dive.edito.eu/project-oceanbench/public/IFS/IFS_20240410.nc", "application/x-netcdf", "27ba6654f705072d35b029359f426790"
   "IFS_20240619.nc", "https://minio.dive.edito.eu/project-oceanbench/public/IFS/IFS_20240619.nc", "application/x-netcdf", "3f3583a1609a6f72f0cfbafa3e574f1c"
   "IFS_20240626.nc", "https://minio.dive.edito.eu/project-oceanbench/public/IFS/IFS_20240626.nc", "application/x-netcdf", "ca813acb62797d26b54494199ebae7a7"
   "IFS_20240605.nc", "https://minio.dive.edito.eu/project-oceanbench/public/IFS/IFS_20240605.nc", "application/x-netcdf", "bde2dd85c77a8aa0a26be312f2b216f1"
   "IFS_20240529.nc", "https://minio.dive.edito.eu/project-oceanbench/public/IFS/IFS_20240529.nc", "application/x-netcdf", "61e0ef48d1d15e9af0290ea5429a9f82"
   "IFS_20240515.nc", "https://minio.dive.edito.eu/project-oceanbench/public/IFS/IFS_20240515.nc", "application/x-netcdf", "696608318a7d4ead1e490c16325abd17"
   "IFS_20240417.nc", "https://minio.dive.edito.eu/project-oceanbench/public/IFS/IFS_20240417.nc", "application/x-netcdf", "00045208de478c8b2ba996ec18951b77"
   "IFS_20241106.nc", "https://minio.dive.edito.eu/project-oceanbench/public/IFS/IFS_20241106.nc", "application/x-netcdf", "dbd2de0a6dc4eba973af9977ad9fd4a7"
   "IFS_20240612.nc", "https://minio.dive.edito.eu/project-oceanbench/public/IFS/IFS_20240612.nc", "application/x-netcdf", "ebf6d01aac38fe5776f0ae5471f25101"
   "IFS_20240124.nc", "https://minio.dive.edito.eu/project-oceanbench/public/IFS/IFS_20240124.nc", "application/x-netcdf", "627a0f8ae7c19a14a5676a1c3fd43a18"
   "IFS_20240925.nc", "https://minio.dive.edito.eu/project-oceanbench/public/IFS/IFS_20240925.nc", "application/x-netcdf", "1bdaec151f95ee3171870302784db64e"
   "IFS_20241211.nc", "https://minio.dive.edito.eu/project-oceanbench/public/IFS/IFS_20241211.nc", "application/x-netcdf", "6b392d677bf6ad552e9594b03acc4dcb"
   "IFS_20241225.nc", "https://minio.dive.edito.eu/project-oceanbench/public/IFS/IFS_20241225.nc", "application/x-netcdf", "9392c414bac07d1d9ccb28e708281c88"
   "IFS_20241204.nc", "https://minio.dive.edito.eu/project-oceanbench/public/IFS/IFS_20241204.nc", "application/x-netcdf", "96c281a4fff1fe7580a8b6929ae0f4e6"
   "IFS_20241127.nc", "https://minio.dive.edito.eu/project-oceanbench/public/IFS/IFS_20241127.nc", "application/x-netcdf", "6b5de3974c67577d8e86cd53c9fc862e"
   "IFS_20240403.nc", "https://minio.dive.edito.eu/project-oceanbench/public/IFS/IFS_20240403.nc", "application/x-netcdf", "013a167e1d7336f76173bdfdb92cebf0"
   "IFS_20240131.nc", "https://minio.dive.edito.eu/project-oceanbench/public/IFS/IFS_20240131.nc", "application/x-netcdf", "8bff0e50a952d0818710270ab241296b"
   "IFS_20240306.nc", "https://minio.dive.edito.eu/project-oceanbench/public/IFS/IFS_20240306.nc", "application/x-netcdf", "577da7856a3ff1269208ba659755806d"
   "IFS_20240313.nc", "https://minio.dive.edito.eu/project-oceanbench/public/IFS/IFS_20240313.nc", "application/x-netcdf", "94e447c15db408fc8fe0a4e6f2cb3544"
   "IFS_20240117.nc", "https://minio.dive.edito.eu/project-oceanbench/public/IFS/IFS_20240117.nc", "application/x-netcdf", "e2361e6bea9d99357cd4f0ef4ddce29a"
   "IFS_20240110.nc", "https://minio.dive.edito.eu/project-oceanbench/public/IFS/IFS_20240110.nc", "application/x-netcdf", "bbc4a367b011c9d35fad4d9d402aba70"
   "IFS_20240214.nc", "https://minio.dive.edito.eu/project-oceanbench/public/IFS/IFS_20240214.nc", "application/x-netcdf", "027c3d740f6545b9d95daf169869769c"
   "IFS_20240703.nc", "https://minio.dive.edito.eu/project-oceanbench/public/IFS/IFS_20240703.nc", "application/x-netcdf", "eca4d9f7fdc6e26bc30d9d8248c9db2d"
   "IFS_20240522.nc", "https://minio.dive.edito.eu/project-oceanbench/public/IFS/IFS_20240522.nc", "application/x-netcdf", "1e9897ca8a156304c95d6f3ecd6aaa44"
   "IFS_20240807.nc", "https://minio.dive.edito.eu/project-oceanbench/public/IFS/IFS_20240807.nc", "application/x-netcdf", "880ca95175f589822d09183d31b330c9"
   "IFS_20240911.nc", "https://minio.dive.edito.eu/project-oceanbench/public/IFS/IFS_20240911.nc", "application/x-netcdf", "bee17b08c9febc4f1a079d070b3005a4"
   "IFS_20240327.nc", "https://minio.dive.edito.eu/project-oceanbench/public/IFS/IFS_20240327.nc", "application/x-netcdf", "2ce5508c15373ea8495891f3d895f69b"
   "IFS_20240424.nc", "https://minio.dive.edito.eu/project-oceanbench/public/IFS/IFS_20240424.nc", "application/x-netcdf", "0d79595f8b8f009600db246fdabede20"
   "IFS_20240207.nc", "https://minio.dive.edito.eu/project-oceanbench/public/IFS/IFS_20240207.nc", "application/x-netcdf", "c6bc0816232ae86dcb6b050bd4df7ad1"
   "IFS_20240724.nc", "https://minio.dive.edito.eu/project-oceanbench/public/IFS/IFS_20240724.nc", "application/x-netcdf", "3212608b0dc4f3f5bd0ac0cc08ed3afe"
   "IFS_20240904.nc", "https://minio.dive.edito.eu/project-oceanbench/public/IFS/IFS_20240904.nc", "application/x-netcdf", "cb9a463f87606ec3ce62e231ca7e9443"
   "IFS_20241218.nc", "https://minio.dive.edito.eu/project-oceanbench/public/IFS/IFS_20241218.nc", "application/x-netcdf", "02aad1d47b8275c4ead5a39fd91c8441"
   "IFS_20241002.nc", "https://minio.dive.edito.eu/project-oceanbench/public/IFS/IFS_20241002.nc", "application/x-netcdf", "eb9822d380cc0d009a6d2f28eb016026"
   "IFS_20240501.nc", "https://minio.dive.edito.eu/project-oceanbench/public/IFS/IFS_20240501.nc", "application/x-netcdf", "0df747a14493825be0705439103f9d78"
   "IFS_20240228.nc", "https://minio.dive.edito.eu/project-oceanbench/public/IFS/IFS_20240228.nc", "application/x-netcdf", "307d692b5e6caa9fc546eb01cf1d9843"
   "IFS_20240918.nc", "https://minio.dive.edito.eu/project-oceanbench/public/IFS/IFS_20240918.nc", "application/x-netcdf", "71ab70a058912fd9440e16b2cbdc3373"
   "IFS_20241023.nc", "https://minio.dive.edito.eu/project-oceanbench/public/IFS/IFS_20241023.nc", "application/x-netcdf", "123db9302091bac1927eb80b2a1f2577"
   "IFS_20240717.nc", "https://minio.dive.edito.eu/project-oceanbench/public/IFS/IFS_20240717.nc", "application/x-netcdf", "e2a1ea87867023fdea824480368cfadc"
   "IFS_20240710.nc", "https://minio.dive.edito.eu/project-oceanbench/public/IFS/IFS_20240710.nc", "application/x-netcdf", "eb3122608256c4190793f1b5929b5ea4"
   "IFS_20240103.nc", "https://minio.dive.edito.eu/project-oceanbench/public/IFS/IFS_20240103.nc", "application/x-netcdf", "a7263adffd01ade25b98f8d46f7bf5e7"
   "IFS_20241120.nc", "https://minio.dive.edito.eu/project-oceanbench/public/IFS/IFS_20241120.nc", "application/x-netcdf", "df0c444915facee4e318e43fdbb229b8"
   "IFS_20240320.nc", "https://minio.dive.edito.eu/project-oceanbench/public/IFS/IFS_20240320.nc", "application/x-netcdf", "7c08fc6b751a70b397601a7192eccd55"
   "IFS_20240821.nc", "https://minio.dive.edito.eu/project-oceanbench/public/IFS/IFS_20240821.nc", "application/x-netcdf", "c833d276511f9fc0da75563c35a8eec2"
   "IFS_20240731.nc", "https://minio.dive.edito.eu/project-oceanbench/public/IFS/IFS_20240731.nc", "application/x-netcdf", "eb266e0ee168054496de84f51c9c4789"
   "IFS_20241016.nc", "https://minio.dive.edito.eu/project-oceanbench/public/IFS/IFS_20241016.nc", "application/x-netcdf", "37629511c24515dbe67aad97c7deb052"
   "IFS_20240828.nc", "https://minio.dive.edito.eu/project-oceanbench/public/IFS/IFS_20240828.nc", "application/x-netcdf", "3ecf5f18018540ce0bbb83b0fb9b322a"
   "IFS_20241009.nc", "https://minio.dive.edito.eu/project-oceanbench/public/IFS/IFS_20241009.nc", "application/x-netcdf", "c8440bfc23775bf72578b7fb3d256a63"
   "IFS_20240508.nc", "https://minio.dive.edito.eu/project-oceanbench/public/IFS/IFS_20240508.nc", "application/x-netcdf", "0c2c41081d4abcdbdaf81593b3c1d46d"
   "GLO12_NOWCAST_20240306.zarr", "https://minio.dive.edito.eu/project-oceanbench/public/GLO12_NOWCAST/20240306.zarr", "application/x-zarr", "n/a"
   "GLO12_NOWCAST_20241225.zarr", "https://minio.dive.edito.eu/project-oceanbench/public/GLO12_NOWCAST/20241225.zarr", "application/x-zarr", "n/a"
   "GLO12_NOWCAST_20240403.zarr", "https://minio.dive.edito.eu/project-oceanbench/public/GLO12_NOWCAST/20240403.zarr", "application/x-zarr", "n/a"
   "GLO12_NOWCAST_20240221.zarr", "https://minio.dive.edito.eu/project-oceanbench/public/GLO12_NOWCAST/20240221.zarr", "application/x-zarr", "n/a"
   "GLO12_NOWCAST_20240821.zarr", "https://minio.dive.edito.eu/project-oceanbench/public/GLO12_NOWCAST/20240821.zarr", "application/x-zarr", "n/a"
   "GLO12_NOWCAST_20240410.zarr", "https://minio.dive.edito.eu/project-oceanbench/public/GLO12_NOWCAST/20240410.zarr", "application/x-zarr", "n/a"
   "GLO12_NOWCAST_20240724.zarr", "https://minio.dive.edito.eu/project-oceanbench/public/GLO12_NOWCAST/20240724.zarr", "application/x-zarr", "n/a"
   "GLO12_NOWCAST_20241120.zarr", "https://minio.dive.edito.eu/project-oceanbench/public/GLO12_NOWCAST/20241120.zarr", "application/x-zarr", "n/a"
   "GLO12_NOWCAST_20241204.zarr", "https://minio.dive.edito.eu/project-oceanbench/public/GLO12_NOWCAST/20241204.zarr", "application/x-zarr", "n/a"
   "GLO12_NOWCAST_20240117.zarr", "https://minio.dive.edito.eu/project-oceanbench/public/GLO12_NOWCAST/20240117.zarr", "application/x-zarr", "n/a"
   "GLO12_NOWCAST_20240710.zarr", "https://minio.dive.edito.eu/project-oceanbench/public/GLO12_NOWCAST/20240710.zarr", "application/x-zarr", "n/a"
   "GLO12_NOWCAST_20240228.zarr", "https://minio.dive.edito.eu/project-oceanbench/public/GLO12_NOWCAST/20240228.zarr", "application/x-zarr", "n/a"
   "GLO12_NOWCAST_20240918.zarr", "https://minio.dive.edito.eu/project-oceanbench/public/GLO12_NOWCAST/20240918.zarr", "application/x-zarr", "n/a"
   "GLO12_NOWCAST_20240207.zarr", "https://minio.dive.edito.eu/project-oceanbench/public/GLO12_NOWCAST/20240207.zarr", "application/x-zarr", "n/a"
   "GLO12_NOWCAST_20240807.zarr", "https://minio.dive.edito.eu/project-oceanbench/public/GLO12_NOWCAST/20240807.zarr", "application/x-zarr", "n/a"
   "GLO12_NOWCAST_20240814.zarr", "https://minio.dive.edito.eu/project-oceanbench/public/GLO12_NOWCAST/20240814.zarr", "application/x-zarr", "n/a"
   "GLO12_NOWCAST_20240103.zarr", "https://minio.dive.edito.eu/project-oceanbench/public/GLO12_NOWCAST/20240103.zarr", "application/x-zarr", "n/a"
   "GLO12_NOWCAST_20240515.zarr", "https://minio.dive.edito.eu/project-oceanbench/public/GLO12_NOWCAST/20240515.zarr", "application/x-zarr", "n/a"
   "GLO12_NOWCAST_20240911.zarr", "https://minio.dive.edito.eu/project-oceanbench/public/GLO12_NOWCAST/20240911.zarr", "application/x-zarr", "n/a"
   "GLO12_NOWCAST_20240320.zarr", "https://minio.dive.edito.eu/project-oceanbench/public/GLO12_NOWCAST/20240320.zarr", "application/x-zarr", "n/a"
   "GLO12_NOWCAST_20240828.zarr", "https://minio.dive.edito.eu/project-oceanbench/public/GLO12_NOWCAST/20240828.zarr", "application/x-zarr", "n/a"
   "GLO12_NOWCAST_20240703.zarr", "https://minio.dive.edito.eu/project-oceanbench/public/GLO12_NOWCAST/20240703.zarr", "application/x-zarr", "n/a"
   "GLO12_NOWCAST_20240619.zarr", "https://minio.dive.edito.eu/project-oceanbench/public/GLO12_NOWCAST/20240619.zarr", "application/x-zarr", "n/a"
   "GLO12_NOWCAST_20240424.zarr", "https://minio.dive.edito.eu/project-oceanbench/public/GLO12_NOWCAST/20240424.zarr", "application/x-zarr", "n/a"
   "GLO12_NOWCAST_20240110.zarr", "https://minio.dive.edito.eu/project-oceanbench/public/GLO12_NOWCAST/20240110.zarr", "application/x-zarr", "n/a"
   "GLO12_NOWCAST_20241127.zarr", "https://minio.dive.edito.eu/project-oceanbench/public/GLO12_NOWCAST/20241127.zarr", "application/x-zarr", "n/a"
   "GLO12_NOWCAST_20240131.zarr", "https://minio.dive.edito.eu/project-oceanbench/public/GLO12_NOWCAST/20240131.zarr", "application/x-zarr", "n/a"
   "GLO12_NOWCAST_20241023.zarr", "https://minio.dive.edito.eu/project-oceanbench/public/GLO12_NOWCAST/20241023.zarr", "application/x-zarr", "n/a"
   "GLO12_NOWCAST_20240612.zarr", "https://minio.dive.edito.eu/project-oceanbench/public/GLO12_NOWCAST/20240612.zarr", "application/x-zarr", "n/a"
   "GLO12_NOWCAST_20240529.zarr", "https://minio.dive.edito.eu/project-oceanbench/public/GLO12_NOWCAST/20240529.zarr", "application/x-zarr", "n/a"
   "GLO12_NOWCAST_20240124.zarr", "https://minio.dive.edito.eu/project-oceanbench/public/GLO12_NOWCAST/20240124.zarr", "application/x-zarr", "n/a"
   "GLO12_NOWCAST_20240925.zarr", "https://minio.dive.edito.eu/project-oceanbench/public/GLO12_NOWCAST/20240925.zarr", "application/x-zarr", "n/a"
   "GLO12_NOWCAST_20241218.zarr", "https://minio.dive.edito.eu/project-oceanbench/public/GLO12_NOWCAST/20241218.zarr", "application/x-zarr", "n/a"
   "GLO12_NOWCAST_20240501.zarr", "https://minio.dive.edito.eu/project-oceanbench/public/GLO12_NOWCAST/20240501.zarr", "application/x-zarr", "n/a"
   "GLO12_NOWCAST_20240417.zarr", "https://minio.dive.edito.eu/project-oceanbench/public/GLO12_NOWCAST/20240417.zarr", "application/x-zarr", "n/a"
   "GLO12_NOWCAST_20240522.zarr", "https://minio.dive.edito.eu/project-oceanbench/public/GLO12_NOWCAST/20240522.zarr", "application/x-zarr", "n/a"
   "GLO12_NOWCAST_20241030.zarr", "https://minio.dive.edito.eu/project-oceanbench/public/GLO12_NOWCAST/20241030.zarr", "application/x-zarr", "n/a"
   "GLO12_NOWCAST_20240717.zarr", "https://minio.dive.edito.eu/project-oceanbench/public/GLO12_NOWCAST/20240717.zarr", "application/x-zarr", "n/a"
   "GLO12_NOWCAST_20240508.zarr", "https://minio.dive.edito.eu/project-oceanbench/public/GLO12_NOWCAST/20240508.zarr", "application/x-zarr", "n/a"
   "GLO12_NOWCAST_20241113.zarr", "https://minio.dive.edito.eu/project-oceanbench/public/GLO12_NOWCAST/20241113.zarr", "application/x-zarr", "n/a"
   "GLO12_NOWCAST_20240605.zarr", "https://minio.dive.edito.eu/project-oceanbench/public/GLO12_NOWCAST/20240605.zarr", "application/x-zarr", "n/a"
   "GLO12_NOWCAST_20241002.zarr", "https://minio.dive.edito.eu/project-oceanbench/public/GLO12_NOWCAST/20241002.zarr", "application/x-zarr", "n/a"
   "GLO12_NOWCAST_20241106.zarr", "https://minio.dive.edito.eu/project-oceanbench/public/GLO12_NOWCAST/20241106.zarr", "application/x-zarr", "n/a"
   "GLO12_NOWCAST_20240904.zarr", "https://minio.dive.edito.eu/project-oceanbench/public/GLO12_NOWCAST/20240904.zarr", "application/x-zarr", "n/a"
   "GLO12_NOWCAST_20241016.zarr", "https://minio.dive.edito.eu/project-oceanbench/public/GLO12_NOWCAST/20241016.zarr", "application/x-zarr", "n/a"
   "GLO12_NOWCAST_20240313.zarr", "https://minio.dive.edito.eu/project-oceanbench/public/GLO12_NOWCAST/20240313.zarr", "application/x-zarr", "n/a"
   "GLO12_NOWCAST_20240214.zarr", "https://minio.dive.edito.eu/project-oceanbench/public/GLO12_NOWCAST/20240214.zarr", "application/x-zarr", "n/a"
   "GLO12_NOWCAST_20241211.zarr", "https://minio.dive.edito.eu/project-oceanbench/public/GLO12_NOWCAST/20241211.zarr", "application/x-zarr", "n/a"
   "GLO12_NOWCAST_20240327.zarr", "https://minio.dive.edito.eu/project-oceanbench/public/GLO12_NOWCAST/20240327.zarr", "application/x-zarr", "n/a"
   "GLO12_NOWCAST_20241009.zarr", "https://minio.dive.edito.eu/project-oceanbench/public/GLO12_NOWCAST/20241009.zarr", "application/x-zarr", "n/a"
   "GLO12_NOWCAST_20240731.zarr", "https://minio.dive.edito.eu/project-oceanbench/public/GLO12_NOWCAST/20240731.zarr", "application/x-zarr", "n/a"
   "GLO12_NOWCAST_20240626.zarr", "https://minio.dive.edito.eu/project-oceanbench/public/GLO12_NOWCAST/20240626.zarr", "application/x-zarr", "n/a"
