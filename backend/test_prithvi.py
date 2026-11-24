from prithvi_service import PrithviService


print("====================================")
print("   SATQUERY - PRITHVI EO TEST")
print("====================================")


IMAGE_PATH = "sample.tif"


prithvi = PrithviService()


print("------------------------------------")
print("Processing satellite image...")
print("------------------------------------")


result = prithvi.extract_features(IMAGE_PATH)


print("------------------------------------")
print("RESULT")
print("------------------------------------")

print("Image processing completed!")

metadata = result["metadata"]

print(f"Image size : {metadata['width']} x {metadata['height']}")
print(f"Bands      : {metadata['bands']}")
print(f"GeoTIFF    : {metadata['is_geotiff']}")
print(f"CRS        : {metadata['crs']}")

if hasattr(result["features"], "shape"):
    print(f"Feature shape: {result['features'].shape}")

print("====================================")
print("   TEST COMPLETED")
print("====================================")