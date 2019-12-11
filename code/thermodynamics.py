import xarray as xr
import numpy as np
import gsw


def sigma0(S, T, p, lat=None, lon=None):
    ones = xr.ones_like(S)
    p = ones * S['p']
    if lat is None:
        lat = ones * S['lat']
    if lon is None:
        lon = ones * S['lon']
    SA = gsw.SA_from_SP(S, p, lon, lat)
    CT = gsw.CT_from_t(SA, T, p)
    rho = gsw.sigma0(SA, CT)
    return rho

def Nsquared(S, T, p, lat=None, lon=None, dim='p'):
    ones = xr.ones_like(S)
    p = ones * S[dim]
    if lat is None:
        lat = ones * S['lat']
    if lon is None:
        lon = ones * S['lon']
    SA = gsw.SA_from_SP(S, p, lon, lat)
    CT = gsw.CT_from_t(SA, T, p)
    N2, pmid = gsw.Nsquared(SA, CT, p, axis=S.get_axis_num(dim))
    return N2, pmid