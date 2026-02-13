from fastapi import FastAPI, UploadFile, File, HTTPException
import requests
import cv2
import numpy as np
import json
import io
from io import BytesIO

app = FastAPI(title="Land Cover Analysis Orchestrator")

# ---------------------------------
# API ENDPOINTS
# ---------------------------------
ENC_API = "https://api-for-ench-detect.onrender.com/extract-plots"
PLOT_API = "https://api-deploy-0ch6.onrender.com/extract-plots"
SEGMENT_API = "https://api-for-map.onrender.com/segment"

# ---------------------------------
# Helper Functions
# ---------------------------------

def forward_to_api(image_bytes, api_url):
    """Sends raw bytes to external APIs"""
    files = {'file': ('image.jpg', image_bytes, 'image/jpeg')}
    response = requests.post(api_url, files=files)
    response.raise_for_status()
    return response

def calculate_unused_area(segmented_img, plot):
    coords = plot["coords"]
    total_pixels = plot["area_pixel"]
    if total_pixels == 0: return 0

    # Create mask for the specific plot
    mask = np.zeros(segmented_img.shape[:2], dtype=np.uint8)
    pts = np.array(coords, np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts], 255)

    # REVISED COLOR MATCHING:
    # Since your segmenter uses fixed colors (Vegetation = [34, 139, 34])
    # RGB [34, 139, 34] in OpenCV BGR is [34, 139, 34].
    # We use a small range to be safe against compression artifacts.
    target_green = np.array([34, 139, 34]) 
    green_mask = cv2.inRange(segmented_img, target_green - 5, target_green + 5)
    
    # Intersect green pixels with the plot mask
    final_mask = cv2.bitwise_and(green_mask, green_mask, mask=mask)
    green_pixels = cv2.countNonZero(final_mask)

    return (green_pixels / total_pixels) * 100

# ---------------------------------
# Main API Endpoint
# ---------------------------------

@app.post("/analyze-land")
async def analyze_land(
    layout_file: UploadFile = File(...), 
    satellite_file: UploadFile = File(...)
):
    try:
        # Read files into memory
        layout_bytes = await layout_file.read()
        sat_bytes = await satellite_file.read()

        # 1. Fetch data from Microservices
        print("Calling Plot API...")
        all_plots = forward_to_api(layout_bytes, PLOT_API).json()

        print("Calling Encroachment API...")
        enc_plots = forward_to_api(sat_bytes, ENC_API).json()

        print("Calling Segmentation API...")
        seg_response = forward_to_api(sat_bytes, SEGMENT_API)
        
        # Decode Segmented Image
        nparr = np.frombuffer(seg_response.content, np.uint8)
        segmented_map = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 2. Filter Encroached Plots
        enc_ids = {plot["id"] for plot in enc_plots}
        valid_plots = [plot for plot in all_plots if plot["id"] not in enc_ids]

        # 3. Generate Report
        report = []
        for plot in valid_plots:
            unused_pct = calculate_unused_area(segmented_map, plot)
            report.append({
                "plot_id": plot["id"],
                "unused_area_percentage": round(unused_pct, 2),
                "center": plot.get("center", [0,0]),
            })

        return {
            "total_plots": len(all_plots),
            "encroached_count": len(enc_ids),
            "valid_plots_count": len(valid_plots),
            "report": report
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
