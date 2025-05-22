import numpy as np
import os, sys, re, math
from scipy.spatial import Delaunay
from shapely.geometry import Point, MultiPoint, MultiLineString, mapping
from shapely.geometry.polygon import Polygon as ShapelyPolygon
from shapely.ops import cascaded_union, polygonize
from matplotlib import path
from shapely.geometry import Point as ShapelyPoint
from geopy.distance import geodesic, great_circle
from geojson import Point as Point_gjson
from geojson import Feature, FeatureCollection, dump

# from decorators import decor_timeit

__author__ = "L.ZAWADZKI and C.REGNIER"
__version__ = 0.2


def alpha_shape(points, alpha):
    """
    Compute the alpha shape (concave hull) of a set
    of points.
    @param points: Iterable container of points.
    @param alpha: alpha value to influence the
        gooeyness of the border. Smaller numbers
        don't fall inward as much as larger numbers.
        Too large, and you lose everything!
    """
    if len(points) < 8:
        print("pts <4")
        # When you have a triangle, there is no sense
        # in computing an alpha shape.
        # poly = MultiPoint(points).convex_hull
        #  return poly
        return MultiPoint(list(points)).convex_hull

    # return MultiPoint(points).convex_hull
    def add_edge(edges, edge_points, coords, i, j):
        """
        Add a line between the i-th and j-th points,
        if not in the list already
        """
        if (i, j) in edges or (j, i) in edges:
            # already added
            return
        edges.add((i, j))
        edge_points.append(coords[[i, j]])

    coords = np.array([point.coords[0] for point in points])
    tri = Delaunay(coords)
    edges = set()
    edge_points = []
    # loop over triangles:
    # ia, ib, ic = indices of corner points of the
    # triangle
    for ia, ib, ic in tri.vertices:
        pa = coords[ia]
        pb = coords[ib]
        pc = coords[ic]
        # Lengths of sides of triangle
        a = math.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
        b = math.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
        c = math.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)
        # Semiperimeter of triangle
        s = (a + b + c) / 2.0
        # Area of triangle by Heron's formula
        area = math.sqrt(s * (s - a) * (s - b) * (s - c))
        if area > 0:
            circum_r = a * b * c / (4.0 * area)
            # Here's the radius filter.
            # print circum_r
            if circum_r < 1.0 / alpha:
                add_edge(edges, edge_points, coords, ia, ib)
                add_edge(edges, edge_points, coords, ib, ic)
                add_edge(edges, edge_points, coords, ic, ia)
    m = MultiLineString(edge_points)
    triangles = list(polygonize(m))
    return cascaded_union(triangles), edge_points


def create_hull(rla_lonmask_tmp, rla_latmask, alpha=0.4):
    """
    Create Shape if missing with cloud hull method
    """
    print("Create hull ")
    pos = np.array(zip(rla_lonmask_tmp, rla_latmask))
    if np.isnan(pos).any():
        ISNA = np.where(np.isnan(pos))
        pos = np.delete(pos, ISNA[0], 0)
    list_points = []
    for x, y in pos:
        x = float(x)
        y = float(y)
        pt = ShapelyPoint(x, y)
        list_points.append(pt)
    concave_hull, edge_points = alpha_shape(list_points, alpha=alpha)
    dict_poly = mapping(concave_hull)
    polygon = ShapelyPolygon(dict_poly["coordinates"][0])
    list_points_simple = []
    posx, posy = polygon.exterior.coords.xy
    for x, y in zip(posx, posy):
        list_points_simple.append((x, y))
    ## Search positions inside polygones
    path_poly = path.Path(list_points_simple)
    return path_poly


# @decor_timeit
def search_points(
    fmt_mask, maskbox, maskval, rla_lonobs, rla_latobs, rla_lonmask_nan, rla_latmask, region, table_hull, **kwargs
):
    """
    Search points inside polygon
    """
    ll_plot = False
    i, j = np.where(maskbox == maskval)
    boxlonmin = np.nanmin(rla_lonmask_nan[i, j])
    boxlonmax = np.nanmax(rla_lonmask_nan[i, j])
    ll_shift = False

    # ~~~~~~~~~ Test the min/max value of the box and shift in 0/360 if necessary ~~~~~~~~

    if boxlonmin < 0 and boxlonmax > 0:
        if boxlonmax > 179.0:

            # ~~~~~~~ Shift 0-360 ~~~~~~~~~~~~
            rla_lonmask_nan_shift = np.where(rla_lonmask_nan < 0, rla_lonmask_nan + 360, rla_lonmask_nan)
            rla_lonmask_tmp = rla_lonmask_nan_shift
            ll_shift = True
        else:
            rla_lonmask_tmp = rla_lonmask_nan
    else:
        rla_lonmask_tmp = rla_lonmask_nan
    boxlonmin = np.nanmin(rla_lonmask_tmp[i, j])
    boxlonmax = np.nanmax(rla_lonmask_tmp[i, j])
    boxlatmin = np.nanmin(rla_latmask[i, j])
    boxlatmax = np.nanmax(rla_latmask[i, j])

    # print (boxlonmin, boxlonmax, boxlatmin, boxlatmax)
    if fmt_mask in ["basin", "box", "basin_pol"]:
        rla_lonobs_tmp = rla_lonobs.copy()
        if ll_shift:
            rla_lonobs_tmp = np.where(rla_lonobs_tmp < 0, rla_lonobs_tmp + 360, rla_lonobs_tmp)
        ibox = np.where(
            (rla_lonobs_tmp >= boxlonmin)
            & (rla_lonobs_tmp < boxlonmax)
            & (rla_latobs >= boxlatmin)
            & (rla_latobs < boxlatmax)
        )[0]

        if np.size(ibox) != 0:
            if fmt_mask in ["basin", "basin_pol"]:
                region = re.sub(" ", "_", region)
                # ~~~~~~~~~~~ Search positions inside polygons ~~~~~~~~~~
                if table_hull is not None:
                    p = table_hull[region]
                else:
                    p = create_hull(rla_lonmask_tmp[i, j], rla_latmask[i, j], alpha=0.4)
            else:
                pos = np.array(zip(rla_lonmask_tmp[i, j], rla_latmask[i, j]))
                if np.isnan(pos).any():
                    ISNA = np.where(np.isnan(pos))
                    pos = np.delete(pos, ISNA[0], 0)
                list_points = []
                for x, y in pos:
                    list_points.append((x, y))

                # ~~ Create polygon with convex_hull algo ~~
                poly = MultiPoint(list_points).convex_hull
                posx, posy = poly.exterior.coords.xy
                list_points_simple = []
                for x, y in zip(posx, posy):
                    list_points_simple.append((x, y))
                p = path.Path(list_points_simple)

            # ~~~~~~~~~~~ Search positions inside polygons ~~~~~~~~~~
            list_input = []
            for x, y in zip(rla_lonobs_tmp[ibox], rla_latobs[ibox]):
                list_input.append((x, y))
            iboxobs = np.where(p.contains_points(list_input))[0]
            if np.size(iboxobs) == 0:
                iobs = iboxobs
                return iobs
            iobs = ibox[iboxobs]
        else:
            iobs = ibox
    else:
        res = abs(rla_lonmask_tmp[0, 1] - rla_lonmask_tmp[0, 0])
        iobs = np.where(
            (rla_lonobs >= boxlonmin - (res / 2.0))
            & (rla_lonobs < boxlonmin + (res / 2.0))
            & (rla_latobs >= boxlatmin - (res / 2.0))
            & (rla_latobs < boxlatmin + (res / 2.0))
        )[0]

    if ll_plot:
        if fmt_mask in ["basin_pol"]:
            if maskval < 20:
                proj = "npstere"
                zone = "Arctic2"
            else:
                proj = "spstere"
                zone = "Antarctic"
        else:
            proj = "cyl"
            zone = "GLO"
        value = rla_lonobs[:].copy()
        rla_lonobs_plot = rla_lonobs[:].copy()
        rla_lonobs_plot = np.where(rla_lonobs_plot > 180, rla_lonobs_plot - 360, rla_lonobs_plot)
        value[:] = 1.0
        lon_inter = 20
        lat_inter = 20
        colormap = "coolwarm"
        titre = "Points in " + str(region) + " for aice with matplotlib meth"
        outputfile = "test/" + str(region) + ".png"
        colormap = "coolwarm"
        cmin = 0
        cmax = 1
        dpi = 75
        font = 16
        dot_size = 2.0
        # Plot_map(rla_lonobs_plot[iobs].tolist(), rla_latobs[iobs].tolist(), zone, lon_inter, lat_inter, proj,\
        #        dpi, font).scatter_val(value[iobs].tolist(), colormap, cmin, cmax, dot_size, titre, outputfile)

    return iobs


# ------------------------------------------------
def dist_sphere(xlam1, yphi1, xlam2, yphi2):
    """
    Calcul des distances sur une sphere en m. Attention,
    un patch permet de verifier que si la longitude passe de 360 a 0
    la distance est calculee de l'Ouest a l'Est et non l'inverse.
    La routine n'est donc pas adaptee pour calculer des grandes distances
    (superieures a 5000 km)
    """
    R = 6371000.0
    rad = np.pi / 180.0
    if np.ndim(xlam1) == 1 and np.ndim(yphi1) == 1 and np.ndim(xlam2) == 1 and np.ndim(yphi2) == 1:
        if len(xlam1) != len(yphi1) or len(xlam2) != len(yphi2):
            print("dist_sphere : problem with shape of the inputs")
            print(" xlam1,%i yphi1,%i xlam2,%i yphi2,%i " % (len(xlam1), len(yphi1), len(xlam2), len(yphi2)))
            sys.exit(1)
    elif np.ndim(xlam1) == 0 and np.ndim(yphi1) == 0 and np.ndim(xlam2) == 1 and np.ndim(yphi2) == 1:
        if len(xlam2) != len(yphi2):
            print("dist_sphere : problem with shape of the inputs")
            print(" xlam1=%f yphi1=%f xlam2,%i yphi2,%i " % (xlam1, yphi1, len(xlam2), len(yphi2)))
            sys.exit(1)
    elif np.ndim(xlam1) == 0 and np.ndim(yphi1) == 0:
        if np.ndim(xlam2) > 1 and np.ndim(yphi2) > 1:
            print("dist_sphere : problem with shape of the inputs")
            print(" xlam1=%f yphi1=%f xlam2,%s yphi2,%s " % (xlam1, yphi1, np.shape(xlam2), np.shape(yphi2)))
            sys.exit(1)
    else:
        print("dist_sphere : problem with shape of the inputs")
        print(" xlam1,%i yphi1,%i xlam2,%i yphi2,%i " % (len(xlam1), len(yphi1), len(xlam2), len(yphi2)))
        sys.exit(1)

    xlam1 = np.array(xlam1)
    xlam2 = np.array(xlam2)
    yphi1 = np.array(yphi1)
    yphi2 = np.array(yphi2)
    if np.ndim(xlam2) == 1:
        xlam2[(xlam2 - xlam1 > 180.0)] -= 360.0
        xlam2[(xlam2 - xlam1 < -180.0)] += 360.0
    elif np.ndim(xlam2) == 0:
        if xlam2 - xlam1 > 180.0:
            xlam2 -= 360.0
        if xlam2 - xlam1 < -180.0:
            xlam2 += 360.0
    # Conversion radians
    lam1 = np.multiply(xlam1, rad)
    lam2 = np.multiply(xlam2, rad)
    phi1 = np.multiply(yphi1, rad)
    phi2 = np.multiply(yphi2, rad)
    zdist = (
        2.0
        * R
        * np.arcsin(
            np.sqrt(np.sin((phi2 - phi1) / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin((lam2 - lam1) / 2.0) ** 2)
        )
    )
    zdist = np.array(zdist)

    return zdist


# ------------------------------------------------
def dist_sphere_NAZ(lon1, lat1, lon2, lat2):
    # Check dimensions:
    size11 = np.ndim(lon1)
    size12 = np.ndim(lat1)
    size21 = len(lon2)
    size22 = len(lat2)
    if size11 != size12 or size11 != 0 or size21 != size22:
        distance = np.nan
    else:
        rad = np.pi / 180.0
        earthRadius = 6371000.0
        oneDeg = earthRadius * rad

        dLon = (lon2 - lon1) * np.cos(rad * (lat1 + lat2) / 2.0)
        dLat = lat2 - lat1

        distance = np.sqrt(dLon**2 + dLat**2) * oneDeg

    return distance


# ------------------------------------------------


def angle_arcs_spheres(xlam0, yphi0, xlam1, yphi1, xlam2, yphi2):
    """
    Calcul des angles entre deux arcs avec meme point de depart sur une sphere.
    Attention, un patch permet de verifier que si la longitude passe de 360 a 0
    l'angle est calculee de l'Est a l'Ouest et non l'inverse.
    Idem pour la latitude.
    La routine n'est donc pas adaptee pour etre appliquee a de grands arcs
    (superieurs a 5000 km)
    """
    rad = np.pi / 180.0
    xlam0 = np.array(xlam0)
    xlam1 = np.array(xlam1)
    xlam2 = np.array(xlam2)
    yhpi0 = np.array(yphi0)
    yhpi1 = np.array(yphi1)
    yhpi2 = np.array(yphi2)
    if np.ndim(xlam1) == 1:
        xlam1[(xlam1 - xlam0 > 180)] -= 360
        xlam1[(xlam1 - xlam0 < -180)] += 360
    elif np.ndim(xlam1) == 0:
        if xlam1 - xlam0 > 180:
            xlam1 -= 360
        if xlam1 - xlam0 < -180:
            xlam1 += 360
    if np.ndim(xlam2) == 1:
        xlam2[(xlam2 - xlam0 > 180)] -= 360
        xlam2[(xlam2 - xlam0 < -180)] += 360
    elif np.ndim(xlam2) == 0:
        if xlam2 - xlam0 > 180:
            xlam2 -= 360
        if xlam2 - xlam0 < -180:
            xlam2 += 360

    lam0 = np.multiply(xlam0, rad)
    lam1 = np.multiply(xlam1, rad)
    lam2 = np.multiply(xlam2, rad)
    phi0 = np.multiply(yphi0, rad)
    phi1 = np.multiply(yphi1, rad)
    phi2 = np.multiply(yphi2, rad)

    signe = np.sign(
        (xlam1 - xlam0) * (yphi2 - yphi0) - (xlam2 - xlam0) * (yphi1 - yphi0)
    )  # Signe du produit vectoriel = signe angle

    alpha_sph = np.zeros_like(xlam1)

    cos0 = np.sqrt(np.sin(phi1) * np.sin(phi2) + np.cos(phi1) * np.cos(phi2) * np.cos(lam2 - lam1))
    cos1 = np.sqrt(np.sin(phi0) * np.sin(phi2) + np.cos(phi0) * np.cos(phi2) * np.cos(lam2 - lam0))
    cos2 = np.sqrt(np.sin(phi0) * np.sin(phi1) + np.cos(phi0) * np.cos(phi1) * np.cos(lam1 - lam0))
    sin1 = np.sqrt(1 - np.sin(phi0) * np.sin(phi2) - np.cos(phi0) * np.cos(phi2) * np.cos(lam2 - lam0))
    sin2 = np.sqrt(1 - np.sin(phi0) * np.sin(phi1) - np.cos(phi0) * np.cos(phi1) * np.cos(lam1 - lam0))

    cos0 = np.array(cos0)
    cos1 = np.array(cos1)
    cos2 = np.array(cos2)
    sin1 = np.array(sin1)
    sin2 = np.array(sin2)

    if np.ndim(xlam0) == 1:
        for il in range(len(xlam0)):
            if sin1[il] == 0:
                alpha_sph[il] = np.arccos(cos2[il]) * signe[il] / rad
            elif sin2[il] == 0:
                alpha_sph[il] = np.arccos(cos1[il]) * signe[il] / rad
            else:
                alpha_sph[il] = (
                    np.arccos((cos0[il] - cos1[il] * cos2[il]) / sin1[il] / sin2[il]) * signe[il] / rad
                )  # angle en degrees
    elif np.ndim(xlam0) == 0:
        if sin1 == 0:
            alpha_sph = np.arccos(cos2) * signe / rad
        elif sin2 == 0:
            alpha_sph = np.arccos(cos1) * signe / rad
        else:
            alpha_sph = np.arccos((cos0 - cos1 * cos2) / sin1 / sin2) * signe / rad  # angle en degrees
    # si on fait l'approximation d'une vue plane pour comparer
    prodscal = (xlam1 - xlam0) * (xlam2 - xlam0) + (yphi2 - yphi0) * (yphi1 - yphi0)
    norm01 = np.sqrt((xlam1 - xlam0) ** 2 + (yphi1 - yphi0) ** 2)
    norm02 = np.sqrt((xlam2 - xlam0) ** 2 + (yphi2 - yphi0) ** 2)
    alpha_2D = np.multiply(np.arccos(prodscal / norm01 / norm02), signe / rad)

    return alpha_sph, alpha_2D


# ------------------------------------------------
def computeCurrent_2points(rla_lon0, rla_lat0, rla_lon1, rla_lat1, rdelta_t):

    rla_hyp = dist_sphere(rla_lon0, rla_lat0, rla_lon1, rla_lat1)
    rla_opp = dist_sphere(rla_lon1, rla_lat0, rla_lon1, rla_lat1)
    nb_output = np.size(rla_hyp)
    rla_rad = []
    rla_alpha = []
    for ji in np.arange(0, nb_output):
        if rla_hyp[ji] != 0:
            rla_rad.append(np.arcsin(abs(rla_opp[ji]) / abs(rla_hyp[ji])))
            if rla_lat0[ji] > rla_lat1[ji]:
                rla_rad[ji] = -rla_rad[ji]
            if rla_lon0 > rla_lon1:
                rla_rad[ji] = np.pi - rla_rad[ji]
            rla_alpha.append(rla_rad[ji] * 180 / np.pi)
            if rla_alpha[ji] < 0:
                rla_alpha[ji] += 360
        else:
            rla_alpha.append(0)

    rla_current = (abs(rla_hyp)) / rdelta_t  # m/s
    rla_zonal = rla_current * np.cos(rla_rad)
    rla_merid = rla_current * np.sin(rla_rad)
    return rla_zonal, rla_merid


# ------------------------------------------------
def direction_trigo_convention_OMM(dlong, dlat):

    # la convention OMM signifie : N=0deg, E=90deg
    # on passe les valeurs suivantes
    # dlat= lat2-lat1
    # dlong= long2-long1

    if np.isfinite(dlong) and np.isfinite(dlat):
        if dlong != 0 and dlat != 0:
            if dlong > 0 and dlat > 0:
                teta = 360 * np.arctan(abs(dlat / dlong)) / (2 * np.pi)  # on prend le 0 comme origine, rotation trigo
                teta = 90 - teta  # si on prend le N=0, E=90
            if dlong < 0 and dlat > 0:
                teta = 180 - 360 * np.arctan(abs(dlat / dlong)) / (
                    2 * np.pi
                )  #  on prend le 0 comme origine, rotation trigo
                teta = np.remainder(
                    3 * 90 + 360 * np.arctan(abs(dlat / dlong)) / (2 * np.pi), 360
                )  # si on prend le N=0, E=90
            if dlong > 0 and dlat < 0:
                teta = 360 - 360 * np.arctan(abs(dlat / dlong)) / (
                    2 * np.pi
                )  # on prend le 0 comme origine, rotation trigo
                teta = np.remainder(
                    90 + 360 * np.arctan(abs(dlat / dlong)) / (2 * np.pi), 360
                )  # si on prend le N=0, E=90
            if dlong < 0 and dlat < 0:
                teta = 180 + 360 * np.arctan(abs(dlat / dlong)) / (
                    2 * np.pi
                )  # on prend le 0 comme origine, rotation trigo
                teta = np.remainder(
                    2 * 90 + (90 - 360 * np.arctan(abs(dlat / dlong)) / (2 * np.pi)), 360
                )  # si on prend le N=0, E=90

        if dlong == 0 and dlat != 0:
            if dlat > 0:
                teta = 90  # on prend le 0 comme origine
                teta = 0  # si on prend le N=0, E=90
            else:
                teta = 270
                teta = 180  # si on prend le N comme origine, E=90

        if dlat == 0 and dlong != 0:
            if dlong > 0:
                teta = 0
                teta = 90  # si on prend le N=0, E=90
            else:
                teta = 180
                teta = np.remainder(270, 360)  # si on prend le N=0, E=90

        if dlong == 0 and dlat == 0:
            teta = np.nan  # pas definie

    else:
        teta = np.nan

    return teta


# --------------------------------------------------------------------------------------------------
def cloudArea(rla_points, surf="plane"):
    """.> Brief:
        Returns the area covered by a 2-dimensional cloud of points.
    If 'surf' is set to 'plane' (default), the units of the
      area is the same as the points units. The area of the polygon
      is computed on a plane surface.
    If 'surf' is set to 'earth', the points coordinates are assumed
      to be longitudes and latitudes and the area of the polygone is
      computed on the surface of the Earth (Radius = 6371000. m )
    .> Usage : area = cloudArea(rla_points,surf='plane')
        where rla_points is an numpy.array with the 2D-coordinates
        of each point. E.g. :
        rla_points =    array([[  x1, y1],
                               [  x2, y2],
                                ...,
                               [ xn,  yn]])
    .> Method : Delaunay triangulation
    (cf. http://fr.wikipedia.org/wiki/Triangulation_de_Delaunay)
    .> Author : lzawadzki
    .> Creation 06.03.2013
    """
    rla_points = np.array(rla_points)
    shape = rla_points.shape
    if len(shape) != 2:
        print("shape of rla_points is incorrect, should be (nb points, 2)")
        return
    if shape[1] != 2:
        print("second dimension of points array should have length 2, got %i" % shape[1])
        return
    if shape[0] < 3:
        print("You need at least 3 points to define a polygone, got %i" % shape[0])
        return

    # rla_points = np.multiply(360,np.random.rand(600, 2))
    # Performs Delaunay triangulation
    delaun = Delaunay(rla_points)

    # Compute area of the polygon
    if surf == "plane":
        # Method : sum the area of each triangle computed with cross product
        area = 0.0
        for tri in delaun.vertices:
            v1 = rla_points[tri[1]] - rla_points[tri[0]]
            v2 = rla_points[tri[2]] - rla_points[tri[0]]
            area += abs(np.cross(v1, v2))
        return area / 2.0
    elif surf == "earth":
        # Method : sum the area of each triangle computed with cross product
        rad = np.pi / 180.0
        area = 0.0
        for tri in delaun.vertices:
            # Let's compute the distances and angles on the sphere,
            # then project it on a plane surface and use cross product with sinus
            xlam0 = rla_points[tri[0]][0]
            xlam1 = rla_points[tri[1]][0]
            xlam2 = rla_points[tri[2]][0]
            yphi0 = rla_points[tri[0]][1]
            yphi1 = rla_points[tri[1]][1]
            yphi2 = rla_points[tri[2]][1]

            c1 = dist_sphere(xlam0, yphi0, xlam1, yphi1)
            c2 = dist_sphere(xlam0, yphi0, xlam2, yphi2)
            a, angl = angle_arcs_spheres(xlam0, yphi0, xlam1, yphi1, xlam2, yphi2)
            area += abs(c1 * c2 * np.sin(angl / rad))
        return area / 2.0
    else:
        print("cloudArea() got an unexpected keyword for argument surf '%s'" % surf)
        return


# --------------------------------------------------------------------------------------------------
def cloudHull(rla_points):
    """
    A partir d'un nuage de points, retourne un ensemble
    de  lignes qui forme un contour convexe
    """
    # rla_points = np.multiply(360,np.random.rand(600, 2))
    tri = Delaunay(rla_points)

    # Make a list of line segments:
    # edge_lines = [ ((x1_1, y1_1), (x2_1, y2_1)),
    #                 ((x1_2, y1_2), (x2_2, y2_2)),
    #                 ... ]
    edge_lines = []
    edge_points = []

    def add_edge(i, j):
        """Add a line between the i-th and j-th points, if not in the list already"""
        edges = set()
        if (i, j) in edges or (j, i) in edges:
            # already added
            return
        edges.add((i, j))
        edge_lines.append(rla_points[[i, j]])
        edge_points.append(rla_points[i])
        edge_points.append(rla_points[j])

    for ia, ib in tri.convex_hull:
        add_edge(ia, ib)

    return edge_lines


# --------------------------------------------------------------------------------------------------


def point_inside_polygon(ra_x, ra_y, poly):
    """
    determine if the point of an array are inside a given polygon or not
    Polygon is a list of (x,y) pairs. But the points must be successively adjacent
    """

    n = len(poly)
    out = np.zeros(len(ra_x))
    i = 0
    for x, y in zip(ra_x, ra_y):
        inside = False
        p1x, p1y = poly[0]
        for i in range(1, n + 1):
            p2x, p2y = poly[i % n]
            if y > min(p1y, p2y) and y <= max(p1y, p2y) and x <= max(p1x, p2x):
                if p1y != p2y:
                    xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                if p1x == p2x or x <= xinters:
                    inside = not inside
            p1x, p1y = p2x, p2y
        out[i] = inside
        i += 1

    return out


# --------------------------------------------------------------------------------------------------


def point_inside_polygon2(ra_x, ra_y, poly):
    """
    Determine if a point is inside a given polygon or not
    Polygon is a list of ([x1,y1],[x2,y2]) pairs. The points [x1,y1]
    and [x2,y2] must be adjacent
    But the successive lines do not have to
    """

    n = len(poly)
    out = np.zeros(len(ra_x))
    i = 0
    for x, y in zip(ra_x, ra_y):
        inside = False
        for line in poly:
            p1x = line[0, 0]
            p1y = line[0, 1]
            p2x = line[1, 0]
            p2y = line[1, 1]
            if y > min(p1y, p2y) and y <= max(p1y, p2y) and x <= max(p1x, p2x):
                if p1y != p2y:
                    xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                if p1x == p2x or x <= xinters:
                    inside = not inside
            p1x, p1y = p2x, p2y
        out[i] = inside
        i += 1

    return out


# --------------------------------------------------------------------------------------------------


def CheckDistance(rla_lon0, rla_lat0, rla_lon1, rla_lat1, dist_max=50.0, name="test"):
    """
    Compute distance between a series of points and test if the coloc is ok
    inputs :
       rla_lon0,rla_lat0: positions of initial points
       rla_lon1,rla_lat1: positions of coloc points
       dist_max: maximum of distance
       name:  name of the mooring/drifter...
    output:
       rla_Dist:  array of distances
    """
    features = []
    rla_lat = np.nan * np.zeros((np.shape(rla_lat1)[0]))
    for i in range(len(rla_lat)):
        if rla_lat1[i] > 90.0:
            print("Latitude > 90.0")
            print(rla_lat1[i])
            print(name)
            rla_lat[i] = 90.0
        else:
            rla_lat[i] = rla_lat1[i]
    if abs(np.nanmax(rla_lat0)) > 90.0 or abs(np.nanmax(rla_lat1)) > 90.0:
        print(name, np.nanmax(rla_lat0), np.nanmax(rla_lat1))

    rla_Dist = [geodesic((rla_lat0[i], rla_lon0[i]), (rla_lat[i], rla_lon1[i])).km for i in range(len(rla_lon0))]
    if np.nanmax(rla_Dist) > dist_max:
        # print ('Prb with coloc :%s' %(name))
        # print (rla_Dist)
        # print ('Modele lon/lat')
        # print (rla_lon0)
        # print (rla_lat0)
        # print ('Drifter lon/lat')
        # print (rla_lon1)
        # print (rla_lat)
        # print ('---------')
        # print (np.nanmax(rla_Dist))
        for i in range(len(rla_lon0)):
            point = Point_gjson((rla_lon1[i], rla_lat[i]))
            features.append(Feature(geometry=point, properties={"name": name}))

    return rla_Dist, features
