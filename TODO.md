Big things:

- Merge visual variety code from 2018 MTAP journal paper into the API.
- Crawler.py to re-create datasets from YFCC100M


Cleanup:

- Documentation and cleanup of stuff in lib/
- Object oriented model for create matrix/eigenvalues stuff
- Streamline model for different datasets
- Restructure the feature extraction in a way that is incremental instead of recalculating all values for overlapping datasets (e.g. when calculating cache for 1000v2000v5000 images it recalculates overlapping parts for every batch instead of just once)
- Do better exception handling.
