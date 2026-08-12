import easyocr
import cv2
import numpy as np
import os
import base64

def process_slide_to_html_standalone(image_path, output_html_path):
    print(f"\n--- Processing {image_path} ---")
    
    # 1. Use EasyOCR for stable coordinates
    reader = easyocr.Reader(['id', 'en'], gpu=True)
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"Error: Could not read {image_path}.")
        return
        
    height, width, _ = img.shape
    results = reader.readtext(image_path)
    
    mask = np.zeros((height, width), dtype=np.uint8)
    html_elements = ""
    
    for (bbox, text, prob) in results:
        top_left = bbox[0]
        bottom_right = bbox[2]
        
        x = int(top_left[0])
        y = int(top_left[1])
        w = int(bottom_right[0] - top_left[0])
        h = int(bottom_right[1] - top_left[1])
        
        # Draw on mask for inpainting (to erase original text)
        pad = 4
        cv2.rectangle(mask, (max(0, x-pad), max(0, y-pad)), (min(width, x+w+pad), min(height, y+h+pad)), 255, -1)
        
        # 2. Extract Exact Text Color using K-Means
        crop = img[max(0, y):min(height, y+h), max(0, x):min(width, x+w)]
        text_color_css = "#ffffff" # Fallback
        
        if crop.size > 0:
            pixels = crop.reshape((-1, 3))
            pixels = np.float32(pixels)
            
            if len(pixels) >= 2:
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
                _, labels, centers = cv2.kmeans(pixels, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
                centers = np.uint8(centers)
                counts = np.bincount(labels.flatten())
                
                # The cluster with FEWER pixels is the text
                text_idx = np.argmin(counts)
                text_color = centers[text_idx]
                text_color_css = f"rgb({text_color[2]}, {text_color[1]}, {text_color[0]})"
            
        font_size = int(h * 0.75)
        
        # 3. Create HTML element with transparent background
        html_elements += f"""
            <div class="extracted-text" 
                 style="left: {x}px; top: {y}px; width: {w}px; height: {h}px; font-size: {font_size}px; color: {text_color_css};" 
                 contenteditable="true">
                 {text}
            </div>
        """
        
    # 4. Inpaint the background to remove text
    print("Applying inpainting to erase original text...")
    clean_bg = cv2.inpaint(img, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    
    # 5. Base64 Encode the Image (Prevents browser loading errors)
    print("Encoding background image to Base64...")
    _, buffer = cv2.imencode('.jpg', clean_bg)
    b64_str = base64.b64encode(buffer).decode('utf-8')
    bg_data_url = f"data:image/jpeg;base64,{b64_str}"
    
    # 6. Construct Final Single-File HTML
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Slide Mockup</title>
        <style>
            body {{ 
                margin: 0; 
                padding: 40px; 
                background-color: #222; 
                display: flex;
                justify-content: center;
                overflow-x: hidden;
            }}
            .slide-container {{
                position: relative;
                width: {width}px;
                height: {height}px;
                /* Base64 image guarantees loading */
                background-image: url('{bg_data_url}');
                background-size: cover;
                background-repeat: no-repeat;
                box-shadow: 0 10px 25px rgba(0,0,0,0.5);
                /* Automatically scales the massive image to fit your screen width */
                transform-origin: top center;
                transform: scale(min(1, calc(90vw / {width})));
            }}
            .extracted-text {{
                position: absolute;
                font-family: 'Inter', 'Segoe UI', Arial, sans-serif;
                font-weight: 700;
                background-color: transparent; /* Seamless blend */
                box-sizing: border-box;
                border: 1px dashed transparent;
                display: flex;
                align-items: center;
                justify-content: center;
                text-align: center;
                line-height: 1.1;
                white-space: pre-wrap;
            }}
            .extracted-text:hover {{
                border: 1px dashed #3b82f6;
                background-color: rgba(255,255,255,0.1);
                cursor: text;
            }}
            .extracted-text:focus {{
                outline: 2px solid #3b82f6;
                background-color: rgba(255,255,255,0.9); /* Makes editing easier */
                color: #000 !important;
                z-index: 10;
            }}
        </style>
    </head>
    <body>
        <div class="slide-container">
            {html_elements}
        </div>
    </body>
    </html>
    """
    
    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
        
    print(f"Success! Standalone HTML saved to {output_html_path}")

if __name__ == "__main__":
    input_folder = "slides_presentation"
    os.makedirs(input_folder, exist_ok=True)
    
    for i in range(1, 6):
        input_image = f"{input_folder}/slide{i}.jpg"
        output_html = f"data/output_slide{i}.html"
        
        if os.path.exists(input_image):
            process_slide_to_html_standalone(input_image, output_html)