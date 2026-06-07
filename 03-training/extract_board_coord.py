import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
img_path = 'test/t4.jpeg'
img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(f"Image not found at path: {img_path}")

# Convert to RGB for display
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Step 1: Grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Step 2: Gaussian Blur
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Step 3: Canny edge detection
edges = cv2.Canny(blur, 50, 150)

# Show intermediate images
plt.figure(figsize=(18, 6))
plt.subplot(1, 3, 1)
plt.imshow(gray, cmap='gray')
plt.title("Grayscale")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(blur, cmap='gray')
plt.title("Gaussian Blur")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(edges, cmap='gray')
plt.title("Canny Edges")
plt.axis('off')
plt.show()

# Step 4: Find contours with hierarchy (for nested contours)
contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

print(f"Total contours found: {len(contours)}")

# Step 5: Filter quadrilateral contours with area > 500
quad_contours = []
areas = []

for c in contours:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    if len(approx) == 4:
        area = cv2.contourArea(c)
        if area > 500:
            quad_contours.append((approx, area))
            areas.append(area)

# Sort contours by area descending
quad_contours = sorted(quad_contours, key=lambda x: x[1], reverse=True)
areas = sorted(areas, reverse=True)

print("Detected quadrilateral contour areas (descending):")
print(areas)

# Show all quadrilateral contours (unfiltered)
all_quads_img = img_rgb.copy()
for cnt, _ in quad_contours:
    cv2.drawContours(all_quads_img, [cnt], -1, (0, 255, 0), 2)  # Green contours

plt.figure(figsize=(10, 8))
plt.imshow(all_quads_img)
plt.title("All Quadrilateral Contours (Area > 500)")
plt.axis('off')
plt.show()

# Step 6: Thresholds for big and medium boards
big_threshold = 20000
medium_threshold = 5000

big_contours_raw = []
medium_contours_raw = []

for approx, area in quad_contours:
    if area >= big_threshold:
        big_contours_raw.append(approx)
    elif medium_threshold <= area < big_threshold:
        medium_contours_raw.append(approx)

print(f"Raw counts: {len(big_contours_raw)} big, {len(medium_contours_raw)} medium")

# Step 7: Define function to check if corners are similar
def are_corners_similar(c1, c2, dist_thresh=15):
    pts1 = c1.reshape(4, 2)
    pts2 = c2.reshape(4, 2)

    pts1_sorted = pts1[np.argsort(pts1.sum(axis=1))]
    pts2_sorted = pts2[np.argsort(pts2.sum(axis=1))]

    distances = np.linalg.norm(pts1_sorted - pts2_sorted, axis=1)
    return np.all(distances < dist_thresh)

# Step 8: Filter duplicate contours by corner similarity
def filter_similar_boxes(boxes, dist_thresh=15):
    filtered = []
    for box in boxes:
        duplicate_found = False
        for kept in filtered:
            if are_corners_similar(box, kept, dist_thresh):
                duplicate_found = True
                break
        if not duplicate_found:
            filtered.append(box)
    return filtered

big_contours = filter_similar_boxes(big_contours_raw)
medium_contours = filter_similar_boxes(medium_contours_raw)

print(f"Filtered counts: {len(big_contours)} big, {len(medium_contours)} medium")

# Step 9: Draw filtered contours with numbering
output_img = img_rgb.copy()

def contour_centroid(cnt):
    M = cv2.moments(cnt)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = cnt[0][0]
    return cx, cy

big_colors = [(255, 0, 255), (200, 0, 200), (255, 100, 255), (180, 0, 180), (230, 50, 230)]
medium_colors = [(0, 0, 255), (0, 100, 255), (100, 100, 255), (50, 50, 255), (0, 150, 255)]

for i, cnt in enumerate(big_contours, 1):
    color = big_colors[(i-1) % len(big_colors)]
    cv2.drawContours(output_img, [cnt], -1, color, 3)
    cx, cy = contour_centroid(cnt)
    cv2.putText(output_img, f"B{i}", (cx - 20, cy), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

for i, cnt in enumerate(medium_contours, 1):
    color = medium_colors[(i-1) % len(medium_colors)]
    cv2.drawContours(output_img, [cnt], -1, color, 3)
    cx, cy = contour_centroid(cnt)
    cv2.putText(output_img, f"M{i}", (cx - 20, cy), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

# Show final image with filtered contours
plt.figure(figsize=(14, 12))
plt.imshow(output_img)
plt.title("Detected Boards with Numbering: Big (B#), Medium (M#), filtered by corner similarity")
plt.axis('off')
plt.show()
