# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""WS column facts over the whole v2 store: coverage and magnitude."""
import os, numpy, xarray
from concurrent.futures import ProcessPoolExecutor

SRC = "/scratch/jseillade/obs-rebuild/store-v2"
UO = "eastward_sea_water_velocity"
VO = "northward_sea_water_velocity"

def day(d):
    p = os.path.join(SRC, d + ".zarr")
    ds = xarray.open_dataset(p, engine="zarr", decode_cf=False, consolidated=True)
    k = ["obs_type","drogued","uo_ws","vo_ws","ws_type",UO,VO,"uo_raw","vo_raw","latitude","longitude","qc_keep"]
    a = {n: ds[n].values for n in k}
    ds.close()
    cur = a["obs_type"] == 3
    kept = cur & (a["qc_keep"] == 1) & numpy.isfinite(a[UO]) & numpy.isfinite(a[VO])
    wsf = numpy.isfinite(a["uo_ws"]) & numpy.isfinite(a["vo_ws"])
    mag = numpy.sqrt(a["uo_ws"]**2 + a["vo_ws"]**2)
    lon = a["longitude"].copy(); lon[lon >= 180.0] -= 360.0
    ibi = (lon >= -19.0) & (lon <= 5.0) & (a["latitude"] >= 26.0) & (a["latitude"] <= 56.0)
    sel = kept & wsf
    out = dict(
        n_cur=int(cur.sum()), n_kept=int(kept.sum()),
        n_kept_wsfinite=int(sel.sum()),
        n_kept_ibi=int((kept & ibi).sum()), n_kept_ibi_ws=int((kept & ibi & wsf).sum()),
        du_sum=float(numpy.nansum(-a["uo_ws"][sel])), dv_sum=float(numpy.nansum(-a["vo_ws"][sel])),
        du_ibi=float(numpy.nansum(-a["uo_ws"][sel & ibi])), dv_ibi=float(numpy.nansum(-a["vo_ws"][sel & ibi])),
        n_ibi_ws=int((sel & ibi).sum()),
        speed_sum=float(numpy.nansum(numpy.sqrt(a[UO][kept]**2 + a[VO][kept]**2))),
    )
    return out, mag[sel].astype("float32"), a["ws_type"][kept].astype("int8"), a["uo_ws"][sel].astype("float32"), a["vo_ws"][sel].astype("float32"), a[UO][sel].astype("float32"), a[VO][sel].astype("float32")

days = sorted(n[:-5] for n in os.listdir(SRC) if n.endswith(".zarr"))
tot = {}
mags, wtypes, uws, vws, uos, vos = [], [], [], [], [], []
with ProcessPoolExecutor(max_workers=12) as ex:
    for out, mag, wt, uw, vw, u, v in ex.map(day, days):
        for k, x in out.items(): tot[k] = tot.get(k, 0) + x
        mags.append(mag); wtypes.append(wt); uws.append(uw); vws.append(vw); uos.append(u); vos.append(v)
mag = numpy.concatenate(mags); wt = numpy.concatenate(wtypes)
uw = numpy.concatenate(uws); vw = numpy.concatenate(vws); u = numpy.concatenate(uos); v = numpy.concatenate(vos)
print("days", len(days))
for k in sorted(tot): print(k, tot[k])
print("coverage kept rows with finite ws:", tot["n_kept_wsfinite"]/tot["n_kept"])
print("coverage IBI:", tot["n_kept_ibi_ws"]/max(tot["n_kept_ibi"],1))
print("ws magnitude m/s: median %.5f p90 %.5f p99 %.5f max %.5f mean %.5f" % (
    numpy.median(mag), numpy.percentile(mag,90), numpy.percentile(mag,99), mag.max(), mag.mean()))
print("mean signed obs delta (=-ws) global: du %.6f dv %.6f" % (tot["du_sum"]/tot["n_kept_wsfinite"], tot["dv_sum"]/tot["n_kept_wsfinite"]))
print("mean signed obs delta IBI: du %.6f dv %.6f n %d" % (tot["du_ibi"]/max(tot["n_ibi_ws"],1), tot["dv_ibi"]/max(tot["n_ibi_ws"],1), tot["n_ibi_ws"]))
print("mean current speed kept: %.5f" % (tot["speed_sum"]/tot["n_kept"]))
print("ws_type counts over kept:", dict(zip(*[x.tolist() for x in numpy.unique(wt, return_counts=True)])))
print("corr(uo_ws, uo) %.4f  corr(vo_ws, vo) %.4f" % (numpy.corrcoef(uw,u)[0,1], numpy.corrcoef(vw,v)[0,1]))
print("rms ws %.5f  rms current %.5f" % (numpy.sqrt((uw**2+vw**2).mean()), numpy.sqrt((u**2+v**2).mean())))
