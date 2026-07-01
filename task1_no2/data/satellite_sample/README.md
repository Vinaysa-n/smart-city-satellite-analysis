Dataset and Data Preprocessing Note

Data Source: TROPOMI Sentinel-5P Nitrogen Dioxide NO2 products.

Temporal Scope: Approximately 600 days of daily satellite observations.

Data Scale: Each raw daily file is around 700 MB, resulting in a total raw data footprint of over 400 GB.

Extraction and Quality Filtering Pipeline:

Pixel Extraction: The processing pipeline automatically parses the raw satellite files to isolate and extract specific pixel values corresponding to the target geographic coordinate of parma.

Quality Control: To ensure data integrity, a strict filtering threshold was implemented. If a target pixel was unavailable due to missing values, extreme cloud cover, or quality flag failures, the data for that specific observation point was systematically rejected and excluded from the dataset.

Output Generation: Validated pixel data was cleaned and exported into individual daily Excel spreadsheets for downstream statistical analysis.