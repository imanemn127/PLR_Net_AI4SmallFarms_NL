// ============================================================
// Google Earth Engine script
// Sentinel-2 NL mosaic 2020 — for PLR-Net dataset reconstruction
//
// Generates a cloud-free median composite over the Netherlands
// for the growing season May–October 2020.
//
// Export: 4 geographic quadrants at 10 m resolution, EPSG:28992.
// Each quadrant is automatically split into sub-tiles by GEE.
// After download, merge all sub-tiles with:
//   gdal_merge.py -o NL_mosaic_2020_10m.tif -co COMPRESS=LZW \
//                 -co BIGTIFF=YES -co TILED=YES S2_NL_2020_NL_*.tif
// ============================================================

// 1. Netherlands boundary (FAO/GAUL is more reliable than LSIB for this country)
var netherlands = ee.FeatureCollection("FAO/GAUL/2015/level0")
  .filter(ee.Filter.eq("ADM0_NAME", "Netherlands"));

// Verification — check in the console before running exports
print("Country name:", netherlands.aggregate_first("ADM0_NAME"));
print("Feature count:", netherlands.size());

// 2. Sentinel-2 Level-2A image collection
// Threshold at 50% to keep enough images for a robust median composite.
// The median compositing step will handle remaining cloud artifacts.
var s2 = ee.ImageCollection("COPERNICUS/S2_SR")
  .filterDate("2020-05-01", "2020-10-01")
  .filterBounds(netherlands)
  .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 50));

print("Number of images after filter:", s2.size());

// 3. Cloud masking using SCL band (Scene Classification Layer, L2A only)
// QA60 is only available in L1C — do not use it with S2_SR.
// SCL class values to mask:
//   3  = cloud shadow
//   8  = cloud, medium probability
//   9  = cloud, high probability
//   10 = thin cirrus
function maskS2clouds(image) {
  var scl = image.select("SCL");
  var mask = scl.neq(3)
               .and(scl.neq(8))
               .and(scl.neq(9))
               .and(scl.neq(10));
  return image.updateMask(mask);
}

// 4. Apply cloud mask and keep only the 4 bands used by PLR-Net
//    B2=Blue, B3=Green, B4=Red, B8=NIR (all at 10 m native resolution)
var s2_masked = s2
  .map(maskS2clouds)
  .select(["B2", "B3", "B4", "B8"]);

// 5. Median composite — robust against remaining cloud artifacts
var mosaic = s2_masked.median().clip(netherlands);

// 6. Preview in the GEE map (RGB visualization only, not exported)
Map.centerObject(netherlands, 8);
Map.addLayer(
  mosaic,
  {bands: ["B4", "B3", "B2"], min: 0, max: 3000},
  "NL Sentinel-2 2020 mosaic"
);

// ============================================================
// EXPORT — 4 geographic quadrants
// The full country (~300 x 350 km at 10 m) exceeds GEE export limits.
// Each quadrant is automatically split into sub-tiles by GEE.
// All sub-tiles share the same CRS and resolution — merge them directly.
// ============================================================

var regions = {
  "NL_NW": ee.Geometry.Rectangle([3.2, 52.3, 5.5, 53.6]),
  "NL_NE": ee.Geometry.Rectangle([5.5, 52.3, 7.3, 53.6]),
  "NL_SW": ee.Geometry.Rectangle([3.2, 50.7, 5.5, 52.3]),
  "NL_SE": ee.Geometry.Rectangle([5.5, 50.7, 7.3, 52.3]),
};

Object.keys(regions).forEach(function(name) {
  Export.image.toDrive({
    image: mosaic.clip(regions[name]),
    description: "S2_NL_2020_" + name,
    folder: "PLRNet_NL",
    fileNamePrefix: "S2_NL_2020_" + name,
    region: regions[name],
    scale: 10,
    crs: "EPSG:28992",
    maxPixels: 1e10,
    fileFormat: "GeoTIFF",
  });
});

print("Export tasks created — click Run in the Tasks tab (top right)");
