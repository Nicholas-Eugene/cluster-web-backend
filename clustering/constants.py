"""
Clustering Application Constants
Contains all constant values used throughout the clustering application
"""

# Column Names - Standard column names for data processing
COLUMN_KABUPATEN_KOTA = "kabupaten_kota"
COLUMN_TAHUN = "tahun"
COLUMN_IPM = "ipm"
COLUMN_GARIS_KEMISKINAN = "garis_kemiskinan"
COLUMN_PENGELUARAN_PER_KAPITA = "pengeluaran_per_kapita"
COLUMN_PROVINSI = "provinsi"
COLUMN_LATITUDE = "latitude"
COLUMN_LONGITUDE = "longitude"

# Required columns for data validation
REQUIRED_COLUMNS = {
    COLUMN_KABUPATEN_KOTA: ["kabupaten_kota", "kabupaten/kota", "kabupaten kota"],
    COLUMN_TAHUN: ["tahun"],
    COLUMN_IPM: ["ipm"],
    COLUMN_GARIS_KEMISKINAN: ["garis_kemiskinan", "garis kemiskinan"],
    COLUMN_PENGELUARAN_PER_KAPITA: [
        "pengeluaran_per_kapita",
        "pengeluaran per kapita",
        "pengeluaran_perkapita",
    ],
}

# Feature columns for clustering
CLUSTERING_FEATURES = [
    COLUMN_IPM,
    COLUMN_GARIS_KEMISKINAN,
    COLUMN_PENGELUARAN_PER_KAPITA,
]

# Algorithm names
ALGORITHM_FCM = "fcm"
ALGORITHM_OPTICS = "optics"
SUPPORTED_ALGORITHMS = [ALGORITHM_FCM, ALGORITHM_OPTICS]

# Algorithm display names
ALGORITHM_DISPLAY_NAMES = {
    ALGORITHM_FCM: "Fuzzy C-Means",
    ALGORITHM_OPTICS: "OPTICS",
}

# Clustering modes
MODE_PER_YEAR = "per_year"
MODE_ALL_YEARS = "all_years"
CLUSTERING_MODES = [MODE_PER_YEAR, MODE_ALL_YEARS]

# Default parameter values for FCM
DEFAULT_NUM_CLUSTERS = 3
DEFAULT_FUZZY_COEFF = 2.0
DEFAULT_MAX_ITER = 300
DEFAULT_TOLERANCE = 0.0001

# Default parameter values for OPTICS
DEFAULT_MIN_SAMPLES = 5
DEFAULT_XI = 0.05
DEFAULT_MIN_CLUSTER_SIZE = 0.05

# Cluster interpretation thresholds (normalized 0-1 scale)
THRESHOLD_LOW = 0.33
THRESHOLD_HIGH = 0.67

# Poverty line ratio thresholds
RATIO_BELOW_POVERTY = 1.0
RATIO_SLIGHTLY_ABOVE = 1.3
RATIO_WELL_ABOVE = 2.0

# File formats
SUPPORTED_FILE_FORMATS = ["csv", "xlsx", "xls"]
EXPORT_FORMATS = ["csv", "json"]

# Noise cluster ID (for OPTICS)
NOISE_CLUSTER_ID = -1

# Matplotlib backend (for headless server environments)
MATPLOTLIB_BACKEND = "Agg"

# Metrics display precision
METRICS_DECIMAL_PLACES = 4
