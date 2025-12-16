import ee
import geopandas as gpd
import folium
import streamlit as st
from shapely.geometry import box
from streamlit_folium import st_folium
from geopy.distance import great_circle
from pyproj import Geod
import numpy as np
import json
import os
from datetime import date
from google.oauth2 import service_account

# ----------------------------------
# Init
# ----------------------------------
# This block handles authentication for both local dev (browser) and cloud deployment (Service Account)
try:
    # Check if the 'EE_SA_KEY' environment variable exists (Deployment Mode)
    if "EE_SA_KEY" in os.environ:
        service_account_info = json.loads(os.environ["EE_SA_KEY"])
        creds = service_account.Credentials.from_service_account_info(service_account_info)
        ee.Initialize(credentials=creds, project="ee-geogrids")
    else:
        # Fallback to default/local authentication (Development Mode)
        ee.Initialize(project="ee-geogrids")
except Exception as e:
    st.error(f"Earth Engine failed to initialize: {e}")
    st.info("If running in the cloud, ensure the 'EE_SA_KEY' environment variable is set with your Service Account JSON.")
    st.stop()

WGS84_GEOD = Geod(ellps="WGS84")

# ----------------------------------
# Helpers
# ----------------------------------

def geodesic_area_km2(gdf: gpd.GeoDataFrame) -> float:
    """Geodesic area (km^2) for WGS84 GeoDataFrame."""
    if gdf is None or gdf.empty:
        return 0.0
    if gdf.crs is None or gdf.crs.to_string() != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")

    area_m2_total = 0.0
    for geom in gdf.geometry:
        if geom is None:
            continue

        if geom.geom_type == "Polygon":
            lon, lat = geom.exterior.coords.xy
            a, _ = WGS84_GEOD.polygon_area_perimeter(lon, lat)
            area_m2_total += abs(a)
            for ring in geom.interiors:
                lon_h, lat_h = ring.coords.xy
                a_h, _ = WGS84_GEOD.polygon_area_perimeter(lon_h, lat_h)
                area_m2_total -= abs(a_h)

        elif geom.geom_type == "MultiPolygon":
            for poly in geom.geoms:
                lon, lat = poly.exterior.coords.xy
                a, _ = WGS84_GEOD.polygon_area_perimeter(lon, lat)
                area_m2_total += abs(a)
                for ring in poly.interiors:
                    lon_h, lat_h = ring.coords.xy
                    a_h, _ = WGS84_GEOD.polygon_area_perimeter(lon_h, lat_h)
                    area_m2_total -= abs(a_h)

    return area_m2_total / 1e6


def create_grid(boundary: gpd.GeoDataFrame, block_size_m: int) -> gpd.GeoDataFrame:
    """Create a lon/lat grid over the boundary bbox using geodesic distances to approximate step counts."""
    minx, miny, maxx, maxy = boundary.total_bounds

    total_x_distance = great_circle((miny, minx), (miny, maxx)).meters
    total_y_distance = great_circle((miny, minx), (maxy, minx)).meters

    num_x_steps = max(1, int(np.ceil(total_x_distance / block_size_m)))
    num_y_steps = max(1, int(np.ceil(total_y_distance / block_size_m)))

    lon_step = (maxx - minx) / num_x_steps
    lat_step = (maxy - miny) / num_y_steps

    cells = []
    for i in range(num_x_steps):
        for j in range(num_y_steps):
            x1 = minx + i * lon_step
            y1 = miny + j * lat_step
            x2 = x1 + lon_step
            y2 = y1 + lat_step
            cells.append(box(x1, y1, x2, y2))

    grid = gpd.GeoDataFrame({"geometry": cells}, crs=boundary.crs)
    if boundary.unary_union is not None:
        grid = grid[grid.intersects(boundary.unary_union)].reset_index(drop=True)
    grid["block_id"] = np.arange(1, len(grid) + 1)
    return grid


# ----------------------------------
# Earth Engine utilities
# ----------------------------------

def _ee_geom_from_gdf(gdf: gpd.GeoDataFrame) -> ee.Geometry:
    """Converts the union of a GeoDataFrame to an Earth Engine geometry."""
    merged = gdf.unary_union
    return ee.Geometry(merged.__geo_interface__)


# ---- Dynamic World (LULC) ----

def get_dynamicworld_mode(boundary: gpd.GeoDataFrame, start_date: str, end_date: str) -> ee.Image:
    """Returns the mode (most common class) of Dynamic World label."""
    ee_geom = _ee_geom_from_gdf(boundary)
    col = (
        ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
        .filterDate(str(start_date), str(end_date))
        .filterBounds(ee_geom)
    )
    # Use mode() for categorical labels to avoid float interpolation issues in visualization
    return col.select("label").mode()


def filter_grid_dynamicworld(grid: gpd.GeoDataFrame, imagery: ee.Image, built_frac_threshold: float) -> gpd.GeoDataFrame:
    # OPTIMIZATION: Convert entire GDF to FC at once
    fc = ee.FeatureCollection(json.loads(grid.to_json())['features'])

    # imagery is the mode image of 'label'. 6 = Built.
    built = imagery.eq(6)
    stats = built.reduceRegions(collection=fc, reducer=ee.Reducer.mean(), scale=10)
    gj = stats.getInfo()

    gdf = gpd.GeoDataFrame.from_features(gj["features"], crs="EPSG:4326")
    if "mean" in gdf.columns:
        gdf.rename(columns={"mean": "built_frac"}, inplace=True)

    gdf = gdf[gdf["built_frac"].fillna(0) >= built_frac_threshold].reset_index(drop=True)
    return gdf


# ---- VIIRS Nighttime Lights ----

def get_viirs_median_image(start_date: str, end_date: str) -> ee.Image:
    col = ee.ImageCollection("NASA/VIIRS/002/VNP46A2").filterDate(str(start_date), str(end_date))
    return col.select("DNB_BRDF_Corrected_NTL").median()


def get_open_buildings_fc(boundary: gpd.GeoDataFrame, min_confidence: float = 0.75) -> ee.FeatureCollection:
    """Return Open Buildings polygons filtered to the sampling boundary and confidence."""
    ee_boundary = _ee_geom_from_gdf(boundary)
    return (
        ee.FeatureCollection("GOOGLE/Research/open-buildings/v3/polygons")
        .filterBounds(ee_boundary)
        .filter(ee.Filter.gte("confidence", float(min_confidence)))
    )


def get_viirs_median(boundary: gpd.GeoDataFrame, start_date: str, end_date: str) -> ee.Image:
    ee_geom = _ee_geom_from_gdf(boundary)
    return get_viirs_median_image(start_date, end_date).clip(ee_geom)


def filter_grid_viirs(grid: gpd.GeoDataFrame, imagery: ee.Image, ntl_threshold: float) -> gpd.GeoDataFrame:
    # OPTIMIZATION: Convert entire GDF to FC at once
    fc = ee.FeatureCollection(json.loads(grid.to_json())['features'])

    stats = imagery.reduceRegions(collection=fc, reducer=ee.Reducer.median(), scale=500)
    gj = stats.getInfo()

    gdf = gpd.GeoDataFrame.from_features(gj["features"], crs="EPSG:4326")
    if "median" in gdf.columns:
        gdf.rename(columns={"median": "ntl_median"}, inplace=True)

    gdf = gdf[gdf["ntl_median"].fillna(0) >= ntl_threshold].reset_index(drop=True)
    return gdf


@st.cache_data(show_spinner=False)
def viirs_ntl_stats(boundary_geom_geojson: dict, start_date: str, end_date: str) -> dict:
    """Return min/p75/max of VIIRS median NTL within boundary geometry."""
    geom = ee.Geometry(boundary_geom_geojson)
    img = get_viirs_median_image(start_date, end_date).clip(geom)

    stats = img.reduceRegion(
        reducer=ee.Reducer.percentile([0, 75, 100]),
        geometry=geom,
        scale=500,
        bestEffort=True,
        maxPixels=1e13,
    )
    d = stats.getInfo() or {}

    return {
        "min": float(d.get("DNB_BRDF_Corrected_NTL_p0", 0.0) or 0.0),
        "p75": float(d.get("DNB_BRDF_Corrected_NTL_p75", 0.0) or 0.0),
        "max": float(d.get("DNB_BRDF_Corrected_NTL_p100", 0.0) or 0.0),
    }


# ---- Open Buildings ----


def filter_grid_open_buildings(grid: gpd.GeoDataFrame, min_buildings: int, boundary: gpd.GeoDataFrame, progress_callback=None) -> gpd.GeoDataFrame:
    """Filter grid cells using Google Open Buildings via Spatial Join with Progress Bar."""
    min_buildings = int(min_buildings)

    # 1) Pre-filter buildings using the boundary
    ee_boundary = _ee_geom_from_gdf(boundary)
    
    # Select only the geometry and empty properties to make the Join lighter
    # convert to centroids to make intersection checks much faster
    bld = ee.FeatureCollection("GOOGLE/Research/open-buildings/v3/polygons") \
        .filterBounds(ee_boundary) \
        .filter(ee.Filter.gte("confidence", 0.75)) \
        .map(lambda f: ee.Feature(f.geometry().centroid()))

    # 2) Define the Spatial Join
    intersect_filter = ee.Filter.intersects(
        leftField='.geo',
        rightField='.geo',
        maxError=10
    )
    
    save_all_join = ee.Join.saveAll(
        matchesKey='matched_buildings'
    )
    
    # 3) Process in Batches
    batch_size = 50 
    results = []
    
    total_batches = (len(grid) + batch_size - 1) // batch_size

    for i in range(0, len(grid), batch_size):
        # Update progress if callback provided
        if progress_callback:
            batch_num = (i // batch_size) + 1
            progress_callback(batch_num, total_batches)

        chunk = grid.iloc[i : i + batch_size]
        
        # Convert chunk to FC
        fc_chunk = ee.FeatureCollection(json.loads(chunk.to_json())['features'])
        
        # Apply Join
        grid_with_matches = save_all_join.apply(fc_chunk, bld, intersect_filter)
        
        # Count matches
        def add_count(feature):
            count = ee.List(feature.get('matched_buildings')).size()
            return feature.set('building_count', count).select(['block_id', 'building_count', '.geo'])

        counted = grid_with_matches.map(add_count)
        
        # Filter server-side
        filtered_chunk = counted.filter(ee.Filter.gte('building_count', min_buildings))
        
        # Fetch results for this chunk
        try:
            chunk_data = filtered_chunk.getInfo()
            if 'features' in chunk_data:
                results.extend(chunk_data['features'])
        except Exception as e:
            print(f"Batch {i} failed: {e}")
            continue

    # 4) Reconstruct GeoDataFrame
    if not results:
        empty = grid.iloc[:0].copy()
        empty["building_count"] = 0
        return empty

    gdf = gpd.GeoDataFrame.from_features(results, crs="EPSG:4326")
    
    # Fix types
    if "block_id" in gdf.columns:
        gdf["block_id"] = gdf["block_id"].astype(int, errors="ignore")
    if "building_count" in gdf.columns:
        gdf["building_count"] = gdf["building_count"].fillna(0).astype(float).astype(int)
    
    return gdf.reset_index(drop=True)


# ----------------------------------
# Map rendering
# ----------------------------------

_DYNAMICWORLD_PALETTE = [
    "#419BDF",  # 0 Water
    "#397D49",  # 1 Trees
    "#88B053",  # 2 Grass
    "#7A87C6",  # 3 Flooded vegetation
    "#E49635",  # 4 Crops
    "#DFC35A",  # 5 Shrub & scrub
    "#C4281B",  # 6 Built area
    "#A59B8F",  # 7 Bare ground
    "#B39FE1",  # 8 Snow & ice
]


def create_folium_map(boundary=None, grid=None, filtered=None, imagery=None, imagery_type=None, buildings_fc=None):

    m = folium.Map(location=[0, 0], zoom_start=2, control_scale=True)

    if boundary is None: 
        return m
    
    if boundary is not None and not boundary.empty:
        b = boundary.to_crs("EPSG:4326")
        minx, miny, maxx, maxy = b.total_bounds
        m.fit_bounds([[miny, minx], [maxy, maxx]])
        folium.GeoJson(
            b.__geo_interface__,
            name="Boundary",
            style_function=lambda x: {"color": "#1f77b4", "weight": 3},
        ).add_to(m)

    if grid is not None and not grid.empty:
        g = grid.to_crs("EPSG:4326")
        folium.GeoJson(
            g.__geo_interface__,
            name="Grid (all)",
            style_function=lambda x: {"color": "#555555", "weight": 1, "fillOpacity": 0},
        ).add_to(m)

    if filtered is not None and not filtered.empty:
        f = filtered.to_crs("EPSG:4326")
        folium.GeoJson(
            f.__geo_interface__,
            name="Grid (filtered)",
            style_function=lambda x: {"color": "#ff7f0e", "weight": 2, "fillOpacity": 0.1},
        ).add_to(m)

    if imagery is not None and imagery_type:
        if imagery_type == "dynamicworld":
            # NOTE: imagery is already selected/reduced to 'label' mode in get_dynamicworld_mode
            # But just in case it has bands, we select 'label' if available, else assume it's single band
            vis = {"min": 0, "max": 8, "palette": _DYNAMICWORLD_PALETTE}
            mid = imagery.getMapId(vis)
            folium.raster_layers.TileLayer(
                tiles=mid["tile_fetcher"].url_format,
                attr="Google Earth Engine",
                name="Dynamic World (label)",
                overlay=True,
                control=True,
                opacity=0.7,
            ).add_to(m)

        elif imagery_type == "viirs":
            vis = {"min": 0, "max": 60}
            mid = imagery.getMapId(vis)
            folium.raster_layers.TileLayer(
                tiles=mid["tile_fetcher"].url_format,
                attr="Google Earth Engine",
                name="VIIRS NTL (median)",
                overlay=True,
                control=True,
                opacity=0.7,
            ).add_to(m)

    # Optional: Open Buildings polygons layer (outline only)
    if buildings_fc is not None:
        try:
            bld_img = buildings_fc.style(**{
                "color": "FFFF00",
                "fillColor": "00000000",
                "width": 1,
            })
            mid = bld_img.getMapId({})
            folium.raster_layers.TileLayer(
                tiles=mid["tile_fetcher"].url_format,
                attr="Google Earth Engine",
                name="Open Buildings (confidence ≥ 0.75)",
                overlay=True,
                control=True,
                opacity=0.9,
            ).add_to(m)
        except Exception:
            # If styling/tiling fails for any reason, skip layer gracefully
            pass

    folium.LayerControl(collapsed=False).add_to(m)
    return m


# ----------------------------------
# Streamlit UI
# ----------------------------------

st.set_page_config(layout="wide")

st.title("GeoGrids")
st.markdown("Create geospatial sampling grids for urban field research.")

# Stable layout: left controls / right map
col1, col2 = st.columns([1, 3], gap="large")

with col1:
    
    # Initialize placeholders
    boundary_geojson = None
    
    # ----------------------------------------------------
    # SECTION 1: Define Sampling Boundary
    # ----------------------------------------------------
    with st.expander("1. Define sampling boundary", expanded=True):
        boundary_option = st.radio(
            "Boundary Source",
            ("Enter bounding box", "Upload GeoJSON"),
            index=0,
            label_visibility="collapsed"
        )

        boundary_input = None
        if boundary_option == "Enter bounding box":
            st.caption("Bounding box in WGS84 (lon/lat).")
            c1, c2 = st.columns(2)
            with c1:
                min_lon = st.number_input("Min longitude", value=-43.2198, format="%.6f")
                min_lat = st.number_input("Min latitude", value=-22.9698, format="%.6f")
            with c2:
                max_lon = st.number_input("Max longitude", value=-43.1599, format="%.6f")
                max_lat = st.number_input("Max latitude", value=-22.914, format="%.6f")
            
            if min_lon < max_lon and min_lat < max_lat:
                boundary_input = gpd.GeoDataFrame(geometry=[box(min_lon, min_lat, max_lon, max_lat)], crs="EPSG:4326")
            else:
                st.warning("Bounding box coordinates are invalid (min must be < max).")
        else:
            f = st.file_uploader("Upload boundary GeoJSON", type=["geojson"])
            if f is not None:
                boundary_input = gpd.read_file(f)
                if boundary_input.crs is None:
                    boundary_input = boundary_input.set_crs("EPSG:4326")
                else:
                    boundary_input = boundary_input.to_crs("EPSG:4326")

        # Validate & persist boundary
        if boundary_input is not None and not boundary_input.empty:
            boundary_input = boundary_input.to_crs("EPSG:4326")
            area_km2 = geodesic_area_km2(boundary_input)
            if area_km2 > 2000:
                st.error(f"Boundary area is {area_km2:,.2f} km² (> 2,000 km²). Please choose a smaller area.")
                boundary_input = None # invalidate
            else:
                st.info(f"Boundary area: {area_km2:,.2f} km² (OK)")
                st.session_state["boundary"] = boundary_input
                # Use unary_union for correct GeoJSON representation of full boundary
                boundary_geojson = boundary_input.unary_union.__geo_interface__
                st.session_state["boundary_geojson"] = boundary_geojson
        else:
             st.session_state["boundary"] = None
             st.session_state["grid"] = None

    # ----------------------------------------------------
    # SECTION 2: Define Grid Cell Size
    # ----------------------------------------------------
    with st.expander("2. Define grid cell size", expanded=True):
        block_size = st.number_input("Block size (m)", min_value=100, value=150, step=10)
        
        # Check for changes or missing grid
        current_boundary = st.session_state.get("boundary")
        current_boundary_geojson = st.session_state.get("boundary_geojson")
        
        # Initialize tracker if needed
        if "last_params" not in st.session_state:
             st.session_state["last_params"] = {"block_size": None, "boundary_geojson": None}
        
        should_regenerate = False
        
        # 1. Check if params changed
        if (block_size != st.session_state["last_params"]["block_size"] or 
            current_boundary_geojson != st.session_state["last_params"]["boundary_geojson"]):
            should_regenerate = True
            
        # 2. Check if grid is missing but boundary exists (e.g. first load)
        if st.session_state.get("grid") is None and current_boundary is not None:
             should_regenerate = True
             
        if should_regenerate:
            if current_boundary is not None:
                g = create_grid(current_boundary, int(block_size))
                st.session_state["grid"] = g
                # Clear filtered results because baseline changed
                st.session_state["filtered"] = None 
                st.session_state["filter_summary"] = None
                
                # Update tracker
                st.session_state["last_params"]["block_size"] = block_size
                st.session_state["last_params"]["boundary_geojson"] = current_boundary_geojson
            else:
                st.session_state["grid"] = None
                
        if st.session_state.get("grid") is None:
             st.caption("Please define a boundary to generate the grid.")

    # ----------------------------------------------------
    # SECTION 3: Filter Grid
    # ----------------------------------------------------
    with st.expander("3. Filter grid", expanded=True):
        filter_method = st.selectbox(
            "Method",
            (
                "No filtering",
                "Dynamic World (built-up fraction)",
                "VIIRS Nighttime lights (median NTL)",
                "Open Buildings (building count)",
            ),
            index=0,
        )

        # Method Change detection to reset persistent summary
        if "last_filter_method" not in st.session_state:
            st.session_state["last_filter_method"] = filter_method

        if st.session_state["last_filter_method"] != filter_method:
            st.session_state["filtered"] = None
            st.session_state["filter_summary"] = None
            st.session_state["last_filter_method"] = filter_method

        # Reset ephemeral state variables
        st.session_state["imagery"] = None
        st.session_state["imagery_type"] = None
        st.session_state["buildings_fc"] = None
        
        # Params containers
        built_frac_threshold = None
        dw_start = dw_end = None
        viirs_start = viirs_end = None
        viirs_threshold = None
        min_buildings = None

        if filter_method == "Dynamic World (built-up fraction)":
            st.caption("Built-up is Dynamic World label == 6; the fraction is the share of built-up pixels per grid cell.")
            built_frac_threshold = st.slider("Built-up threshold (fraction)", 0.0, 1.0, 0.30, 0.01)
            
            # DEFAULT DATES: Last year 2023 for stable default
            default_dw = [date(2023, 1, 1), date(2024, 1, 1)]
            dw_dates = st.date_input("Dynamic World timeframe (start, end)", default_dw)
            st.caption("The app will calculate the most common land cover class (mode) observed across this time range.")
            
            # Auto-preview Imagery
            if isinstance(dw_dates, (list, tuple)) and len(dw_dates) == 2:
                dw_start, dw_end = dw_dates[0].isoformat(), dw_dates[1].isoformat()
                if st.session_state.get("boundary") is not None:
                     # Use mode for better categorical visualization
                     imagery = get_dynamicworld_mode(st.session_state["boundary"], dw_start, dw_end)
                     st.session_state["imagery"] = imagery
                     st.session_state["imagery_type"] = "dynamicworld"

        elif filter_method == "VIIRS Nighttime lights (median NTL)":
            st.caption("Uses NASA/VIIRS/002/VNP46A2 median composite; per-cell median (DNB_BRDF_Corrected_NTL) is compared to a threshold.")
            
            # DEFAULT DATES
            default_viirs = [date(2023, 1, 1), date(2024, 1, 1)]
            viirs_dates = st.date_input("VIIRS timeframe (start, end)", default_viirs)
            st.caption("The app will calculate the median nighttime light intensity observed across this time range.")
            
            if isinstance(viirs_dates, (list, tuple)) and len(viirs_dates) == 2:
                viirs_start, viirs_end = viirs_dates[0].isoformat(), viirs_dates[1].isoformat()
                
                # Auto-preview Imagery
                if st.session_state.get("boundary") is not None:
                    imagery = get_viirs_median(st.session_state["boundary"], viirs_start, viirs_end)
                    st.session_state["imagery"] = imagery
                    st.session_state["imagery_type"] = "viirs"

                    # Calculate stats for slider (this might take a moment)
                    boundary_geojson = st.session_state.get("boundary_geojson")
                    if boundary_geojson:
                         # We don't want this to block the map update too long, but it's needed for the slider
                         with st.spinner("Computing VIIRS NTL stats for slider..."):
                            stats = viirs_ntl_stats(boundary_geojson, viirs_start, viirs_end)
                         
                         vmin, vp75, vmax = float(stats["min"]), float(stats["p75"]), float(stats["max"])
                         if vmax <= vmin:
                             viirs_threshold = st.number_input("NTL threshold (median)", value=float(vp75))
                         else:
                             viirs_threshold = st.slider(
                                "NTL threshold (median DNB_BRDF_Corrected_NTL)",
                                min_value=vmin,
                                max_value=vmax,
                                value=min(max(vp75, vmin), vmax),
                            )

        elif filter_method == "Open Buildings (building count)":
            st.caption("Counts Open Buildings polygons intersecting each grid cell (GOOGLE/Research/open-buildings/v3/polygons).")
            min_buildings = st.number_input("Minimum buildings per cell", min_value=0, value=10, step=1)
            
            # Auto-preview Buildings
            _b = st.session_state.get("boundary")
            if _b is not None and not _b.empty:
                st.session_state["buildings_fc"] = get_open_buildings_fc(_b, min_confidence=0.75)

        else:
            st.caption("No filtering will be applied; all grid cells inside the boundary will be kept.")

        
        # Filter Action
        filter_clicked = st.button("Filter grid", use_container_width=True)

        if filter_clicked:
            b = st.session_state.get("boundary")
            g = st.session_state.get("grid")

            if b is None or g is None:
                st.error("Please define a boundary and grid first.")
            else:
                filtered_result = None

                if filter_method == "No filtering":
                    filtered_result = g

                elif filter_method == "Dynamic World (built-up fraction)":
                    if not (dw_start and dw_end):
                        st.error("Please select a Dynamic World start and end date.")
                    else:
                        with st.spinner("Filtering grid..."):
                            # Re-fetch imagery to be safe/clean inside the logic
                            img = get_dynamicworld_mode(b, dw_start, dw_end)
                            filtered_result = filter_grid_dynamicworld(g, img, float(built_frac_threshold))

                elif filter_method == "VIIRS Nighttime lights (median NTL)":
                    if not (viirs_start and viirs_end):
                        st.error("Please select a VIIRS start and end date.")
                    else:
                        with st.spinner("Filtering grid..."):
                            img = get_viirs_median(b, viirs_start, viirs_end)
                            filtered_result = filter_grid_viirs(g, img, float(viirs_threshold))

                elif filter_method == "Open Buildings (building count)":
                    # Custom progress handling
                    progress_bar = st.progress(0, text="Starting Open Buildings processing...")
                    
                    def update_progress(batch_num, total_batches):
                        pct = min(batch_num / total_batches, 1.0)
                        progress_bar.progress(pct, text=f"Processing batch {batch_num} of {total_batches}...")

                    filtered_result = filter_grid_open_buildings(g, int(min_buildings), b, progress_callback=update_progress)
                    progress_bar.empty()

                st.session_state["filtered"] = filtered_result

                # Calculate Summary and persist it
                try:
                    total_cells = len(g)
                    selected_cells = len(filtered_result) if filtered_result is not None else 0
                    if total_cells > 0:
                        pct = 100.0 * selected_cells / total_cells
                        
                        # Calculate Area
                        area_km2 = 0.0
                        if filtered_result is not None and not filtered_result.empty:
                            area_km2 = geodesic_area_km2(filtered_result)
                        
                        summary_msg = f"Grid created: {total_cells:,} cells. Selected after filtering: {selected_cells:,} ({pct:.1f}%). Total area: {area_km2:.2f} km²."
                        st.session_state["filter_summary"] = summary_msg
                except Exception:
                    st.session_state["filter_summary"] = None
        
        # Display Persistent Summary if exists
        if st.session_state.get("filter_summary"):
            st.success(st.session_state["filter_summary"])

    # Download final grid
    if "filtered" in st.session_state and st.session_state["filtered"] is not None and not st.session_state["filtered"].empty:
        st.download_button(
            label="Download final grid (GeoJSON)",
            data=st.session_state["filtered"].to_json(),
            file_name="sampling_grid.geojson",
            mime="application/json",
            use_container_width=True,
        )


with col2:
    boundary = st.session_state.get("boundary")
    grid = st.session_state.get("grid")
    filtered = st.session_state.get("filtered")
    imagery = st.session_state.get("imagery")
    imagery_type = st.session_state.get("imagery_type")
    buildings_fc = st.session_state.get("buildings_fc")

    m = create_folium_map(boundary=boundary, grid=grid, filtered=filtered, imagery=imagery, imagery_type=imagery_type, buildings_fc=buildings_fc)
    if m is None:
        # Fallback: always render at least a base map
        m = folium.Map(location=[0, 0], zoom_start=2, control_scale=True)

    st_folium(m, height=800, use_container_width=True)