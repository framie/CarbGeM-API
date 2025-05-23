import cv2
import math
import numpy as np
import os
import sys

# Sanitize filename by preventing unwanted characters
def sanitize_filename(filename):
    return "".join( x for x in filename if (x.isalnum() or x in "._- "))


# Format circles from cv.HoughCircles() for cv2 drawing usage
def round_and_format_circles(circles):
    return np.uint16(np.around(circles))


# Detect and return Agar Plate circle within image
def get_agar_plate(gray_image):
    gray_blurred = cv2.medianBlur(gray_image, 31)
    image_height, image_width = gray_blurred.shape[:2]
    max_radius = int(min(image_height, image_width) // 2)
    agar_plates = cv2.HoughCircles(
        gray_blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max_radius,
        param1=100,
        param2=30,
        minRadius=0,
        maxRadius=max_radius
    )
    if agar_plates is None or not agar_plates.any():
        return None
    agar_plates = round_and_format_circles(agar_plates[0])
    plate_x, plate_y, plate_r = max(agar_plates, key=lambda c: c[2])
    return (plate_x, plate_y, plate_r)


# Write Agar Plate info to the top left of image
def write_agar_plate_info(image, agar_plate):
    plate_x, plate_y, plate_r = agar_plate
    plate_diameter_px = plate_r * 2
    px_per_mm = plate_diameter_px / 100
    cv2.circle(image, (plate_x, plate_y), plate_r, (255, 0, 0), 4)
    cv2.putText(image, "Agar Plate", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(image, f"100mm, { plate_diameter_px }px", (30, 76),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(image, f"1mm = { px_per_mm }px", (30, 102),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


# Detect and return Antibiotic Disk circles within image
def get_antibiotic_disks(gray_image):
    gray_blurred = cv2.medianBlur(gray_image, 13)
    antibiotic_disks = cv2.HoughCircles(
        gray_blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=100,
        param1=50,
        param2=60,
        minRadius=20,
        maxRadius=200
    )
    if antibiotic_disks is None or not antibiotic_disks.any():
        return []
    return round_and_format_circles(antibiotic_disks[0])


# Sort Antibiotic Disks from an image into clockwise order
def clockwise_sort(image, antibiotic_disks):
    image_height, image_width = image.shape[:2]
    center_x = image_width / 2
    center_y = image_height / 2

    def clockwise_angle(x, y):
        angle = math.atan2(y - center_y, x - center_x)
        adjusted = (angle + math.pi / 2 + 2 * math.pi) % (2 * math.pi)
        return adjusted

    return sorted(antibiotic_disks, key=lambda c: clockwise_angle(c[0] + c[2], c[1]))


# Ray tracing function to detect Zone of Inhibition surrounding Antibiotic disk
def raytrace_zoi(gray_image, center, start_radius, max_radius=150, step=1, angle_step=5):
    gray_blurred = cv2.medianBlur(gray_image, 11)
    cx, cy = center
    detected_radii = []
    threshold=155
    confirm_count=3

    for angle in range(0, 360, angle_step):
        rad = math.radians(angle)
        consecutive_dark = 0

        for r in range(start_radius, max_radius, step):
            x = int(cx + r * math.cos(rad))
            y = int(cy + r * math.sin(rad))

            if not (0 <= x < gray_blurred.shape[1] and 0 <= y < gray_blurred.shape[0]):
                break

            pixel_val = gray_blurred[y, x]
            if pixel_val < threshold:
                consecutive_dark += 1
                if consecutive_dark >= confirm_count:
                    detected_radii.append(r - (confirm_count - 1))
                    break
            else:
                consecutive_dark = 0

    if detected_radii:
        return int(np.median(detected_radii))
    return None


# Write 
def write_image_to_output(image, input_path, output_folder):
    image_filename, extension = os.path.splitext(input_path.split('/')[-1])
    output_path = f"{output_folder}/{image_filename}{extension}"
    os.makedirs(output_folder, exist_ok=True)
    write_success =  cv2.imwrite(output_path, image)
    if not write_success:
        raise RuntimeError(f"Failed to write image to: {output_path}")


def main():
    input_path = sys.argv[1]
    output_folder = "output"
    print(f"Reading input file { input_path }...")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Image not found: {input_path}")

    image = cv2.imread(input_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    print("Identifying agar plate...")
    agar_plate = get_agar_plate(gray_image)
    if not agar_plate:
        print("No agar plate found in Image.")
        return
    print(f"Found agar plate { agar_plate }")

    output_image = image.copy()
    write_agar_plate_info(output_image, agar_plate)
    plate_diameter_px = agar_plate[2] * 2
    px_per_mm = plate_diameter_px / 100

    print("Identifying antibiotic disks...")
    antibiotic_disks = get_antibiotic_disks(gray_image)
    if len(antibiotic_disks) == 0:
        print("No antibiotic disks found in Image.")
        return
    print(f"Found antibiotic disks { antibiotic_disks }")

    print("Sorting and processing disks...")
    antibiotic_disks = clockwise_sort(image, antibiotic_disks)
    for i, (x, y, r) in enumerate(antibiotic_disks):
        cv2.putText(output_image, f"Disk {i + 1}", (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        outer_radius = raytrace_zoi(gray_image, (x, y), start_radius=r + 4, max_radius=int(plate_diameter_px // 2))
        if outer_radius:
            cv2.circle(output_image, (x, y), outer_radius, (0, 0, 255), 2)
            diameter_mm = (outer_radius * 2) / px_per_mm
            cv2.putText(output_image, f"{diameter_mm:.1f} mm", (x, y + 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    print("Writing image to output...")
    write_image_to_output(output_image, input_path, output_folder)

if __name__ == '__main__':
    main()
