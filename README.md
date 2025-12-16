GeoGrids: Geospatial Sampling Grid Generator

GeoGrids is a Streamlit application designed to streamline the creation of geospatial sampling grids for field surveys and remote sensing analysis. It integrates Google Earth Engine (GEE) to filter grid cells based on real-time satellite imagery and building data.

🚀 Features

Custom Sampling Boundaries: Define your area of interest by drawing a bounding box or uploading a GeoJSON file.

Flexible Grid Generation: Create grids with custom cell sizes (in meters) automatically projected over your boundary.

Advanced Filtering Methods:

Dynamic World (LULC): Filter cells based on the percentage of "Built-up" land cover.

VIIRS Nighttime Lights: Select highly illuminated areas using median nighttime light intensity.

Open Buildings: Filter cells that contain a specific minimum number of buildings (using Google Open Buildings v3).

Interactive Visualization: Real-time map preview of boundaries, grids, filtered results, and underlying satellite imagery layers.

Data Export: Download the final filtered grid as a GeoJSON file for use in GIS software or field data collection tools (e.g., ODK, SurveyCTO).

🛠️ Prerequisites

To run this application locally, you will need:

Python 3.9+

Google Earth Engine Account: You must have a GEE account and a cloud project set up. Sign up here.

Gcloud CLI (Recommended): For authenticating your local environment with Google Earth Engine.

📦 Installation

Clone the repository:

git clone [https://github.com/your-username/geogrids.git](https://github.com/your-username/geogrids.git)
cd geogrids


Create a virtual environment (optional but recommended):

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install dependencies:

pip install -r requirements.txt


(Note: Create a requirements.txt file with: streamlit, geopandas, earthengine-api, folium, streamlit-folium, shapely, geopy, pyproj, numpy)

Authenticate Earth Engine:
Run the following command in your terminal and follow the instructions to link your account.

earthengine authenticate


⚙️ Configuration

Open the main script geogrids_fixed.py and locate the initialization block:

try:
    ee.Initialize(project="ee-geogrids") # Replace 'ee-geogrids' with your GEE Project ID
except Exception as e:
    ...


Replace "ee-geogrids" with your actual Google Cloud Project ID enabled for Earth Engine.

▶️ Usage

Run the Streamlit app:

streamlit run geogrids_fixed.py


Workflow

Define Boundary: Use the "Define sampling boundary" section to enter coordinates or upload a GeoJSON file of your study area.

Set Grid Size: Adjust the "Block size" in meters. The base grid will generate automatically.

Filter: Expand the "Filter grid" section.

Select a method (e.g., Open Buildings).

Adjust thresholds (e.g., Min buildings per cell).

Click Filter grid.

Export: Once satisfied with the filtered map (orange cells), click Download final grid (GeoJSON).

📊 Data Sources

Land Use/Land Cover: Google Dynamic World V1

Nighttime Lights: NASA/VIIRS/002/VNP46A2

Building Footprints: Google Open Buildings V3

📝 License

MIT License