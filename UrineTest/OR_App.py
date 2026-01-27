import os
import json
import math
import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths
from colormath.color_objects import LabColor
from colormath.color_diff import delta_e_cie2000

# Declare as globals
test_img_path = None
#ref_img_path = None
final_img_result_path = None
test_lib_path = None

_SEND_ASYNC = None
_encoder_task = None
DEBUGGING = False


def set_sender(send_async_fn):
    """
    send_async_fn must be an async function: await send_async_fn("msgType~payload")
    """
    global _SEND_ASYNC
    _SEND_ASYNC = send_async_fn


async def clearGlobalReferences():
    global _encoder_task
    if _encoder_task is not None:
        _encoder_task.cancel()
        _encoder_task = None

def deskew_and_crop(img):
    def calculate_skew_angle(roi_edges, width):
        lines = cv2.HoughLinesP(roi_edges, 1, np.pi / 360, threshold=50,
                                minLineLength=int((width / 2) * 0.35), maxLineGap=10)
        angles = []
        if lines is not None:
            for x1, y1, x2, y2 in lines[:, 0]:
                dx = x2 - x1
                dy = y2 - y1
                if dx == 0 or abs(dx) < 10:
                    continue
                theta = math.degrees(math.atan2(dy, dx))
                if -10 < theta < 10:  # Filter near-horizontal lines
                    angles.append(theta)
        return np.median(angles) if angles else 0.0

    def rotate_image(img, angle):
        (h, w) = img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    def crop_largest_contour(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
            margin = 10
            x_crop = max(0, x - margin)
            y_crop = max(0, y - margin)
            w_crop = min(img.shape[1], x + w + margin) - x_crop
            h_crop = min(img.shape[0], y + h + margin) - y_crop
            return img[y_crop:y_crop + h_crop, x_crop:x_crop + w_crop]
        return img

    # Preprocess
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Define ROI
    h, w = img.shape[:2]
    roi_start_y = int(h * 0.30)
    roi_end_y = int(h * 0.65)
    roi_for_edges = blurred[roi_start_y:roi_end_y, :]

    # Detect edges and calculate skew
    edges_in_roi = cv2.Canny(roi_for_edges, 15, 50, apertureSize=3)
    angle = calculate_skew_angle(edges_in_roi, w)

    if angle != 0.0:
        print(f"Detected skew angle = {angle:.2f}°")
    else:
        print("No suitable horizontal lines found for deskewing. Using original image.")

    # Rotate and crop
    rotated = rotate_image(img, angle)
    cropped = crop_largest_contour(rotated)
    #plt.imshow(cropped)
    #plt.show()
    return cropped, angle

def roiextraction(cropped) :
   height,width= cropped.shape[:2]
   half_height= height//2
   img= cropped[:half_height-27,25:]
   img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
   gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
   std_dev = np.std(gray)
   #print(std_dev)
# Check if gray values are uniform
   if std_dev < 5:
    print("Image is completely uniform (blank).")
    raise ValueError("Image is blank")
   else:
    print("The image is not blank.")
   # Compute Laplacian
   laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var() 
   #print('laplacian_var')
   #print(laplacian_var)
   if laplacian_var < 150:
       print('Image is blurred')
       raise ValueError("Image is blurred")
   else:
       print('The image is not blurred')
   #plt.imshow(gray)
   #plt.title('gray')
   #plt.show()
   # Average pixel values along the y-axis (vertical axis)
   avg_along_y = np.mean(gray, axis=1)
   # Average pixel values along the x-axis (horizontal)
   avg_along_x = np.mean(gray, axis=0)


    # Plot both results
   #plt.figure(figsize=(12, 6))


   #plt.subplot(2, 1, 1)
   #plt.plot(avg_along_y, color='blue')
   #plt.title('Average Pixel Intensity Along Y-axis')
   #plt.xlabel('X-axis (Columns)')
   #plt.ylabel('Average Intensity')
   #plt.grid(True)


   #plt.subplot(2, 1, 2)
   #plt.plot(avg_along_x, color='green')
   #plt.title('Average Pixel Intensity Along X-axis')
   #plt.xlabel('Y-axis (Rows)')
   #plt.ylabel('Average Intensity')
   #plt.grid(True)
   #plt.tight_layout()
   #plt.show()
                                                   
   # Find columns and rows to keep
   cols_to_keep = np.where(avg_along_y > 50)[0]
   rows_to_keep = np.where(avg_along_x > 16)[0]
   #print(cols_to_keep)
   #print(rows_to_keep)

   if len(cols_to_keep) == 0 or len(rows_to_keep) == 0:
     print('No strip found')
     raise ValueError("No region found matching the specified average intensity conditions.")


   # Crop image
   y_min, y_max = cols_to_keep[0], cols_to_keep[-1]
   x_min, x_max = rows_to_keep[0], rows_to_keep[-1]
   cropped_img = img[y_min:y_max, x_min:x_max]
   #plt.imshow(cropped_img)
   #plt.title('cropped_img')
   #plt.show()
   height,width= cropped_img.shape[:2]
   #print(f"height:{height},width:{width}")
   if height >23 :
    cropped_img= cropped_img[-20:-2,:]
   
   # Display cropped image
   #cv2.imshow('Cropped Image', cropped_img)
   #cv2.waitKey(0)
   #cv2.destroyAllWindows()
   
   
         
   return cropped_img 

def clean_signals(img, spike_thresh=15, max_spike_width=3):
    """
    Compute mean R, G, B and gray (Y) signals across columns and remove spikes.
    """
    B_line = np.mean(img[:,:,0], axis=0)
    G_line = np.mean(img[:,:,1], axis=0)
    R_line = np.mean(img[:,:,2], axis=0)
    gray   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    Y_line = np.mean(gray, axis=0)

    R_nb = remove_spikes_neighbor(R_line, spike_thresh=spike_thresh, max_spike_width=max_spike_width)
    G_nb = remove_spikes_neighbor(G_line, spike_thresh=spike_thresh, max_spike_width=max_spike_width)
    B_nb = remove_spikes_neighbor(B_line, spike_thresh=spike_thresh, max_spike_width=max_spike_width)
    Y_nb = remove_spikes_neighbor(Y_line, spike_thresh=spike_thresh, max_spike_width=max_spike_width)

    return R_nb, G_nb, B_nb, Y_nb, R_line, B_line, G_line, Y_line


def remove_spikes_neighbor(sig, spike_thresh=15, max_spike_width=3):

    sig = np.asarray(sig, float)
    bad = np.zeros_like(sig, bool)
    deltas = np.abs(np.diff(sig, prepend=sig[0]))
    bad[deltas > spike_thresh] = True
    
    cleaned = sig.copy()
    i, n = 0, len(sig)
    while i < n:
        if not bad[i]:
            i += 1; continue
        j = i
        while j < n and bad[j]:
            j += 1
        if (j - i) <= max_spike_width:
            x0, x1 = max(i-1,0), min(j, n-1)
            y0, y1 = cleaned[x0], cleaned[x1]
            for k in range(i, j):
                t = (k - x0) / (x1 - x0)
                cleaned[k] = (1-t)*y0 + t*y1
        i = j
    return cleaned

def compute_highlighted_derivative(sig, deriv_thresh):
    """
    Given a 1-D de-spiked signal `sig`, return the raw derivative `d`
   and the thresholded derivative `d_high` (values below thresh→0).
    """
    d = np.diff(sig, prepend=sig[0])
    d_high = np.where(np.abs(d) >= deriv_thresh, d, 0)

    return d, d_high

def find_and_merge_peaks(R_nb, G_nb, B_nb, Y_nb, height_thresh=5, merge_distance=6):

    deriv_thresh = 3
    R_d, R_dh = compute_highlighted_derivative(R_nb, deriv_thresh)
    G_d, G_dh = compute_highlighted_derivative(G_nb, deriv_thresh)
    B_d, B_dh = compute_highlighted_derivative(B_nb, deriv_thresh)
    Y_d, Y_dh = compute_highlighted_derivative(Y_nb, deriv_thresh)

    R_abs_d = np.abs(R_dh)
    G_abs_d = np.abs(G_dh)
    B_abs_d = np.abs(B_dh)
    Y_abs_d = np.abs(Y_dh)

    RGB_max_abs_d = np.maximum.reduce([R_abs_d, G_abs_d, B_abs_d, Y_abs_d])
    
    raw_peaks, _ = find_peaks(RGB_max_abs_d, height=height_thresh)
    merged_peaks = []
    groups = []
    current_group = [raw_peaks[0]]
    for i in range(1, len(raw_peaks)):
        if raw_peaks[i] - raw_peaks[i - 1] <= merge_distance:
            current_group.append(raw_peaks[i])
        else:
            groups.append(current_group)
            current_group = [raw_peaks[i]]
    if current_group:
        groups.append(current_group)

    for i, current_group in enumerate(groups):
        group_diffs = np.diff(current_group)
        if np.all(group_diffs <= 4):
            best_peak = int(np.mean(current_group))
        else:
            if i + 1 < len(groups):
                next_group_first = groups[i + 1][0]
                distances_to_next = [abs(p - next_group_first) for p in current_group]
                best_idx   = int(np.argmin(distances_to_next))
                best_peak  = current_group[best_idx]
            else:
                forward_distances = [
                    current_group[j + 1] - current_group[j]
                    if j + 1 < len(current_group) else float('inf')
                    for j in range(len(current_group))
                ]
                best_idx = int(np.argmax(forward_distances))
                best_peak = current_group[best_idx]
        merged_peaks.append(best_peak)

    blank_ranges = [] 
    pad_ranges = []

    image_width = test_img.shape[1]

    i = 0
    expecting = None

    while i < len(merged_peaks) - 1:
        x0 = merged_peaks[i]
        x1 = merged_peaks[i + 1]
        width = x1 - x0

        if expecting is None:
            if 7 <= width <= 16:
                blank_ranges.append((x0, x1))
                expecting = "pad"
                i += 1
            elif 22 <= width <= 40:
                pad_ranges.append((x0, x1))
                expecting = "blank"
                i += 1
            else:
                i += 1
            continue

        if expecting == "pad":
            found = False
            if 22 <= width <= 45:
                pad_ranges.append((x0, x1))
                expecting = "blank"
                i += 1
            else:
                for j in range(i+1, len(merged_peaks)):
                  width = merged_peaks[j] - x0
                  if 22 <= width <= 35:
                    pad_ranges.append((x0, merged_peaks[j]))
                    expecting = "blank"
                    i = j   # jump i forward
                    found = True
                    break
                if not found:
                    i += 1

        elif expecting == "blank":
            if 7 <= width <= 40:
                blank_ranges.append((x0, x1))
                expecting = "pad"
                i += 1
            else:
                if i + 2 < len(merged_peaks):
                    x2 = merged_peaks[i + 2]
                    merged_width = x2 - x0
                    if 7 <= merged_width <= 21:
                        blank_ranges.append((x0, x2))
                        expecting = "pad"
                        i += 2
                    else:
                        i += 1
                else:
                    i += 1

    if len(merged_peaks) > 0:
        last_peak = merged_peaks[-1]
        end_width = image_width - last_peak
        if expecting == "pad" and 21 <= end_width <= 45:
            pad_ranges.append((last_peak, image_width))
        elif expecting == "blank" and 7 <= end_width <= 20:
            blank_ranges.append((last_peak, image_width))


    offset = 180
    peak_coords = [x + offset for x in merged_peaks]

    pad_peak_pairs = []
    for x0, x1 in pad_ranges:
        x0_orig = x0 + offset
        x1_orig = x1 + offset
        pad_peak_pairs.append((x0_orig, x1_orig))

    #print("Merged Peak Coordinates (with offset):", peak_coords)
    #print("Pad Regions:", pad_peak_pairs)

    return merged_peaks, pad_ranges, blank_ranges

def store_pad_regions(img, pad_ranges, offset=0, merged_peaks=None, output_dir='pad_crops'):
    """
    Save pad regions from image.
    If the last peak is in a blank region (not part of any pad_range), it is excluded from cropping logic.
    """
    image_width = img.shape[1]
    #plt.imshow(img)
    #plt.title("Img")
    #plt.show()
    merged_peaks_offset = [p + offset for p in merged_peaks] if merged_peaks else []

    # Check if the last peak is in any valid pad_range (after offset)
    if merged_peaks_offset:
        last_peak = merged_peaks_offset[-1]
        in_valid_range = any(x0 + offset <= last_peak <= x1 + offset for x0, x1 in pad_ranges)

        if not in_valid_range:
            print(f"Last peak {last_peak} is in blank region. Excluding it.")
            merged_peaks_offset.pop()  # Remove last peak
        else:
            print(f"Last peak {last_peak} is within a valid pad region. Keeping it.")

    pad_peak_pairs = []
    pad_colours= []
    all_crops= []
    for i, (x0, x1) in enumerate(pad_ranges):
        x0_orig = x0 + offset
        x1_orig = x1 + offset
        pad_peak_pairs.append((x0_orig, x1_orig))
        print(pad_peak_pairs)

    # Save cropped regions
    os.makedirs(output_dir, exist_ok=True)
    for i, (x0, x1) in enumerate(pad_peak_pairs):
        if x1 > x0:
            crop = img[:, x0:x1]
            filename = os.path.join(output_dir, f'pad_{i}.png')
            #plt.imshow(crop)
            #plt.show()
            #plt.imshow(crop)
            #plt.show()
            crop_centre=crop[5:15,10:20]
            #plt.imshow(crop_centre)
            #plt.show()
            all_crops.append(crop_centre)
            #r,g,b= cv2.split(crop_centre)
            #avg_r= np.mean(r)
            #avg_g= np.mean(g)
            #avg_b=np.mean(b)
            avg_color = cv2.mean(crop_centre)[:3] 
            pad_colours.append(avg_color)
            cv2.imwrite(filename, crop)
            print(f"Saved pad_{i}.png: x = {x0 + offset} to {x1 + offset}")
            print(pad_colours)
            print(len(pad_colours))
            print(f"Saved pad_{i}.png: x = {x0 + offset} to {x1 + offset}")
        else:
            print(f"Skipped pad_{i}: invalid region x = {x0} to {x1}")
    
    return pad_peak_pairs, merged_peaks_offset, pad_colours, all_crops
def rgbtolab(pad_colours):
    #print('pad_colours')
    #print(pad_colours)
    bgr_array = np.array([[ [b, g, r] for r, g, b in pad_colours ]], dtype=np.float32)
    # Step 3: Normalize to 0–1 range for OpenCV if needed (some versions require this for float32)
    #print(bgr_array)
    bgr_array /= 255.0
    #print(bgr_array)
# Convert to Lab
    lab_array = cv2.cvtColor(bgr_array, cv2.COLOR_BGR2Lab)
   
    #print(pad_colours)
    #print(lab_array)
# Flatten to get Lab values in a list
    lab_colours = [tuple(map(float, lab_array[0][i])) for i in range(len(pad_colours))]
    #print(lab_colours)
# Print result
    for i, (rgb, lab) in enumerate(zip(pad_colours, lab_colours)):
        print(f"Color {i + 1} - RGB: {rgb} -> Lab: {lab}")
    
    
    return lab_colours
def bgcomp(lab_colours,data):
       bg_value={}
       deltaE_bg=[]
       deltaE_bg_list=[]
       for patchdetail in data['pads'][0]['patchdetails']:
         bg_value[patchdetail['bloodvalue']] = np.array(patchdetail['L,a,bvalues'], dtype=np.float32) 
       print('bg')
       for key, value in bg_value.items():
          if len(value) >= 3:  # or value.shape[0] >= 3 if it's a NumPy array
            L1, a1, b1 = value[0], value[1], value[2]  # Access elements directly
            L2, a2, b2 = lab_colours[9]
            #print(L2,a2,b2)
            color1 = LabColor(lab_l=L1, lab_a=a1, lab_b=b1)
            color2 = LabColor(lab_l=L2, lab_a=a2, lab_b=b2)

         # Compute Delta E 2000
            deltaE_bg = float(delta_e_cie2000(color1, color2))
            #print(f"Delta E 2000: {deltaE_bg:.2f}")
            deltaE_bg_list.append(deltaE_bg)
       min_bg= min(deltaE_bg_list)
       min_index_bg = deltaE_bg_list.index(min_bg)
       #print(min_index_bg)
       #print(deltaE_bg_list)
       #print(len(deltaE_bg_list))
       blood  = data['pads'][0]["patchdetails"][min_index_bg]["bloodvalue"]
       #print(blood)

       return blood
def bilirubincomp(lab_colours,data):
       bilirubin_value={}
       deltaE_bilirubin=[]
       deltaE_bilirubin_list=[]
       for patchdetail in data['pads'][1]['patchdetails']:
         bilirubin_value[patchdetail['bilirubinvalue']] = np.array(patchdetail['L,a,bvalues'], dtype=np.float32) 
         #print('bilirubin')
       for key, value in bilirubin_value.items():
          if len(value) >= 3:  # or value.shape[0] >= 3 if it's a NumPy array
            L1, a1, b1 = value[0], value[1], value[2]  # Access elements directly
            L2, a2, b2 = lab_colours[8]
            #print(L2,a2,b2)
            color1 = LabColor(lab_l=L1, lab_a=a1, lab_b=b1)
            color2 = LabColor(lab_l=L2, lab_a=a2, lab_b=b2)

         # Compute Delta E 2000
            deltaE_bilirubin = float(delta_e_cie2000(color1, color2))
            #print(f"Delta E 2000: {deltaE_bilirubin:.2f}")
            deltaE_bilirubin_list.append(deltaE_bilirubin)
       min_bilirubin= min(deltaE_bilirubin_list)
       min_index_bilirubin = deltaE_bilirubin_list.index(min_bilirubin)
       #print(min_index_bilirubin)
       #print(deltaE_bilirubin_list)
       #print(len(deltaE_bilirubin_list))
       bilirubin  = data['pads'][1]["patchdetails"][min_index_bilirubin]["bilirubinvalue"]
       #print(bilirubin)

       return bilirubin
def urocomp(lab_colours,data):
      uro_value={}
      deltaE_uro=[]
      deltaE_uro_list=[]
      for patchdetail in data['pads'][2]['patchdetails']:
        uro_value[patchdetail['urobilinogenvalue']] = np.array(patchdetail['L,a,bvalues'], dtype=np.float32) 
      #print('uro')
      for key, value in uro_value.items():
        if len(value) >= 3:  # or value.shape[0] >= 3 if it's a NumPy array
           L1, a1, b1 = value[0], value[1], value[2]  # Access elements directly
           L2, a2, b2 = lab_colours[7]
           #print(L2,a2,b2)
           color1 = LabColor(lab_l=L1, lab_a=a1, lab_b=b1)
           color2 = LabColor(lab_l=L2, lab_a=a2, lab_b=b2)

         # Compute Delta E 2000
           deltaE_uro = float( delta_e_cie2000(color1, color2))
           #print(f"Delta E 2000: {deltaE_uro:.2f}")
           deltaE_uro_list.append(deltaE_uro)
      min_uro= min(deltaE_uro_list)
      min_index_uro = deltaE_uro_list.index(min_uro)
      #print(min_index_uro)
      #print(f"Delta E 2000: {deltaE_uro:.2f}")
      #print(deltaE_uro_list)
      #print(len(deltaE_uro_list))
      urobilinogen  = data['pads'][2]["patchdetails"][min_index_uro]["urobilinogenvalue"]
      #print(urobilinogen)

      return urobilinogen
def ketonecomp(lab_colours,data):
      ketone_value={}
      deltaE_ketone=[]
      deltaE_ketone_list=[]
      for patchdetail in data['pads'][3]['patchdetails']:
        ketone_value[patchdetail['ketonesvalues']] = np.array(patchdetail['L,a,bvalues'], dtype=np.float32) 
      #print('ketone')
      for key, value in ketone_value.items():
         if len(value) >= 3:  # or value.shape[0] >= 3 if it's a NumPy array
           L1, a1, b1 = value[0], value[1], value[2]  # Access elements directly
           L2, a2, b2 = lab_colours[6]
           #print(L2,a2,b2)
           color1 = LabColor(lab_l=L1, lab_a=a1, lab_b=b1)
           color2 = LabColor(lab_l=L2, lab_a=a2, lab_b=b2)

         # Compute Delta E 2000
           deltaE_ketone = float(delta_e_cie2000(color1, color2))
           #print(f"Delta E 2000: {deltaE_ketone:.2f}")
           deltaE_ketone_list.append(deltaE_ketone)
      min_ketone= min(deltaE_ketone_list)
      min_index_ketone = deltaE_ketone_list.index(min_ketone)
      #print(min_index_ketone)
      #print(f"Delta E 2000: {deltaE_ketone:.2f}")
      #print(deltaE_ketone_list)
      #print(len(deltaE_ketone_list))
      ketones  = data['pads'][3]["patchdetails"][min_index_ketone]["ketonesvalues"]
      #print(ketones)

      return ketones
def proteincomp(lab_colours,data):
       pro_value={}
       deltaE_pro=[]
       deltaE_pro_list=[]
       for patchdetail in data['pads'][4]['patchdetails']:
        pro_value[patchdetail['proteinvalue']] = np.array(patchdetail['L,a,bvalues'], dtype=np.float32) 
       #print('pro') 
       for key, value in pro_value.items():
         if len(value) >= 3:  # or value.shape[0] >= 3 if it's a NumPy array
           L1, a1, b1 = value[0], value[1], value[2]  # Access elements directly
           L2, a2, b2 = lab_colours[5]
           #print(L2,a2,b2)
           color1 = LabColor(lab_l=L1, lab_a=a1, lab_b=b1)
           color2 = LabColor(lab_l=L2, lab_a=a2, lab_b=b2)

         # Compute Delta E 2000
           deltaE_pro = float(delta_e_cie2000(color1, color2))
           #print(f"Delta E 2000: {deltaE_pro:.2f}")
           deltaE_pro_list.append(deltaE_pro)
       min_pro= min(deltaE_pro_list)
       min_index_pro = deltaE_pro_list.index(min_pro)
       #print(min_index_pro)
       #print(deltaE_pro_list)
       #print(len(deltaE_pro_list))
       protein  = data['pads'][4]["patchdetails"][min_index_pro]["proteinvalue"]
       #print(protein)

       return protein
def nitritecomp(lab_colours,data):
      nit_value={}
      deltaE_nit=[]
      deltaE_nit_list=[]
      for patchdetail in data['pads'][5]['patchdetails']:
        nit_value[patchdetail['nitritesvalue']] = np.array(patchdetail['L,a,bvalues'], dtype=np.float32) 
      #print('nit')
      for key, value in nit_value.items():
         if len(value) >= 3:  # or value.shape[0] >= 3 if it's a NumPy array
           L1, a1, b1 = value[0], value[1], value[2]  # Access elements directly
           L2, a2, b2 = lab_colours[4]
           #print(L2,a2,b2)
           color1 = LabColor(lab_l=L1, lab_a=a1, lab_b=b1)
           color2 = LabColor(lab_l=L2, lab_a=a2, lab_b=b2)

         # Compute Delta E 2000
           deltaE_nit = float( delta_e_cie2000(color1, color2))
           #print(f"Delta E 2000: {deltaE_nit:.2f}")
           deltaE_nit_list.append(deltaE_nit)
      min_nit= min(deltaE_nit_list)
      min_index_nit = deltaE_nit_list.index(min_nit)
      #print(min_index_nit)
      #print(deltaE_nit_list)
      #print(len(deltaE_nit_list))
      nitrites = data['pads'][5]["patchdetails"][min_index_nit]["nitritesvalue"]
      #print(nitrites)

      return nitrites
def glucosecomp(lab_colours,data):
       glucose_value={}
       deltaE_glucose=[]
       deltaE_glucose_list=[]
       for patchdetail in data['pads'][6]['patchdetails']:
        glucose_value[patchdetail['glucosevalue']] = np.array(patchdetail['L,a,bvalues'], dtype=np.float32) 
       #print('glucose')
       for key, value in glucose_value.items():
          if len(value) >= 3:  # or value.shape[0] >= 3 if it's a NumPy array
            L1, a1, b1 = value[0], value[1], value[2]  # Access elements directly
            L2, a2, b2 = lab_colours[3]
            #print(L2,a2,b2)
            L1= L1.item()
            a1 = a1. item()
            b1 = b1.item()
            #print(pad_colours[3])
            #print(L1,a1,b1)
           # Convert to Python float explicitly (safe even if already float)
            L1, a1, b1 = float(L1), float(a1), float(b1)
            L2, a2, b2 = float(L2), float(a2), float(b2)
             # Calculate delta E
           #color1 = LabColor(lab_l=L1, lab_a=a1, lab_b=b1)
           #color2 = LabColor(lab_l=L2, lab_a=a2, lab_b=b2)
           #color1_np = np.array([L1, a1, b1], dtype=np.float32)
           #color2_np = np.array([L2, a2, b2], dtype=np.float32)
           #color2_np = tuple(int(x) for x in color2_np)
           #print(color1_np)
           #print(color2_np)
           #color2_np = np.array([color2_np], dtype=np.uint8)
           #print(color2_np)
           # Step 1: Create LabColor objects from your Lab values
            lab1 = LabColor(lab_l=L1, lab_a=a1, lab_b=b1)
            lab2 = LabColor(lab_l=L2, lab_a=a2, lab_b=b2)

         # Compute Delta E 2000
            deltaE_glucose = (delta_e_cie2000(lab1, lab2))
            deltaE_glucose_list.append(deltaE_glucose)
       min_glucose= min(deltaE_glucose_list)
       min_index_glucose = deltaE_glucose_list.index(min_glucose)
       #print(min_index_glucose)
       #print(f"Delta E 2000: {deltaE_glucose:.2f}")
       #print(deltaE_glucose_list)
       #print(len(deltaE_glucose_list))
       glucose  = data['pads'][6]["patchdetails"][min_index_glucose]["glucosevalue"]
       #print(glucose)
       return glucose 
def phcomp(lab_colours,data):
       ph_value={}
       deltaE_ph=[]
       deltaE_ph_list=[]
       for patchdetail in data['pads'][7]['patchdetails']:
         ph_value[patchdetail['pHvalue']] = np.array(patchdetail['L,a,bvalues'], dtype=np.float32) 
         #print('pH')
       for key, value in ph_value.items():
         if len(value) >= 3:  # or value.shape[0] >= 3 if it's a NumPy array
           L1, a1, b1 = value[0], value[1], value[2]  # Access elements directly
           L2, a2, b2 = lab_colours[2]
           #print(L2,a2,b2)
           color1 = LabColor(lab_l=L1, lab_a=a1, lab_b=b1)
           color2 = LabColor(lab_l=L2, lab_a=a2, lab_b=b2)

         # Compute Delta E 2000
           deltaE_ph = float(delta_e_cie2000(color1, color2))
           #print(f"Delta E 2000: {deltaE_ph:.2f}")
           deltaE_ph_list.append(deltaE_ph)
       min_ph= min(deltaE_ph_list)
       min_index_ph = deltaE_ph_list.index(min_ph)
       #print(min_index_ph)
       #print(deltaE_ph_list)
       #print(len(deltaE_ph_list))
       pH  = data['pads'][7]["patchdetails"][min_index_ph]["pHvalue"]
       #print(pH)
       return pH
def sgcomp(lab_colours,data):
       sg_value={}
       deltaE_sg=[]
       deltaE_sg_list=[]
       for patchdetail in data['pads'][8]['patchdetails']:
        sg_value[patchdetail['specificgravityvalue']] = np.array(patchdetail['L,a,bvalues'], dtype=np.float32) 
       #print('sg')
       for key, value in sg_value.items():
         if len(value) >= 3:  # or value.shape[0] >= 3 if it's a NumPy array
           L1, a1, b1 = value[0], value[1], value[2]  # Access elements directly
           L2, a2, b2 = lab_colours[1]
           #print(L2,a2,b2)
           color1 = LabColor(lab_l=L1, lab_a=a1, lab_b=b1)
           color2 = LabColor(lab_l=L2, lab_a=a2, lab_b=b2)

         # Compute Delta E 2000
           deltaE_sg = float(delta_e_cie2000(color1, color2))
           #print(f"Delta E 2000: {deltaE_sg:.2f}")
           deltaE_sg_list.append(deltaE_sg)
       min_sg= min(deltaE_sg_list)
       min_index_sg = deltaE_sg_list.index(min_sg)
       #print(min_index_sg)
       #print(deltaE_sg_list)
       #print(len(deltaE_sg_list))
       specificgravity  = data['pads'][8]["patchdetails"][min_index_sg]["specificgravityvalue"]
       #print(specificgravity)
       return specificgravity
         
def leukocytescomp(lab_colours,data):
      leu_value={}
      deltaE_leu=[]
      deltaE_leu_list=[]
      for patchdetail in data['pads'][9]['patchdetails']:
        leu_value[patchdetail['leukocytesvalue']] = np.array(patchdetail['L,a,bvalues'], dtype=np.float32) 
      #print('leu')
      for key, value in leu_value.items():
         if len(value) >= 3:  # or value.shape[0] >= 3 if it's a NumPy array
           L1, a1, b1 = value[0], value[1], value[2]  # Access elements directly
           L2, a2, b2 = lab_colours[0]
           #print(L2,a2,b2)
           color1 = LabColor(lab_l=L1, lab_a=a1, lab_b=b1)
           color2 = LabColor(lab_l=L2, lab_a=a2, lab_b=b2)

         # Compute Delta E 2000
           deltaE_leu = float(delta_e_cie2000(color1, color2))
           #print(f"Delta E 2000: {deltaE_leu:.2f}")
           deltaE_leu_list.append(deltaE_leu)
      min_leu= min(deltaE_leu_list)
      min_index_leu = deltaE_leu_list.index(min_leu)
      #print(min_index_leu)
      #print(deltaE_leu_list)
      #print(len(deltaE_leu_list))
      leukocytes  = data['pads'][9]["patchdetails"][min_index_leu]["leukocytesvalue"]
      #print(leukocytes)
      return leukocytes
 
def finalimageformation(all_crops):
      
      #for i, img in enumerate(all_crops):
          #plt.imshow(img)
          #plt.show()
      final_image = np.ones((200, 10, 3), dtype=np.uint8) * 255
      final_image[0:10,0:10]= all_crops[9]
      final_image[20:30,0:10]= all_crops[8]
      final_image[40:50,0:10]= all_crops[7]
      final_image[60:70,0:10]= all_crops[6]
      final_image[80:90,0:10]= all_crops[5]
      final_image[100:110,0:10]= all_crops[4]
      final_image[120:130,0:10]= all_crops[3]
      final_image[140:150,0:10]= all_crops[2]
      final_image[160:170,0:10]= all_crops[1]
      final_image[180:190,0:10]= all_crops[0]
      #plt.imshow(final_image)
      #plt.show()
      #cv2.imwrite('final_image.png', final_image)
      final_image = cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR)
      cv2.imwrite(final_img_result_path, final_image)
        
      image = cv2.imread(final_img_result_path)
      # Rotate 90 degrees clockwise to make it horizontal
      rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
      
      resized_image = cv2.resize(rotated_image, (474, 25))  # (width, height)

      cv2.imwrite(final_img_result_path, resized_image)

      return final_image  

def plot_diagnostics(
    R_line, G_line, B_line,
    R_nb,   G_nb,   B_nb,
    merged_peaks,
    blank_ranges,
    pad_ranges,
    img           # the cropped strip image (BGR)
):
    """
    4-panel diagnostic:
      0) raw vs cleaned signals
      1) highlighted substantial changes
      2) abs derivative + blank/pad regions + peaks
      3) the strip image
    """

    deriv_thresh = 3
    R_d,  R_dh = compute_highlighted_derivative(R_nb, deriv_thresh)
    G_d,  G_dh = compute_highlighted_derivative(G_nb, deriv_thresh)
    B_d,  B_dh = compute_highlighted_derivative(B_nb, deriv_thresh)
    # Y channel derivative if you need it…

    R_abs_d      = np.abs(R_dh)
    G_abs_d      = np.abs(G_dh)
    B_abs_d      = np.abs(B_dh)
    RGB_max_abs_d = np.maximum.reduce([R_abs_d, G_abs_d, B_abs_d])

    fig, axs = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

    # 0: Raw vs Cleaned
    axs[0].plot(R_line, label='Red Raw',    alpha=0.8)
    axs[0].plot(G_line, label='Green Raw')
    axs[0].plot(B_line, label='Blue Raw')
    axs[0].plot(R_nb,   '--', label='Red Cleaned')
    axs[0].plot(G_nb,   '--', label='Green Cleaned')
    axs[0].plot(B_nb,   '--', label='Blue Cleaned')
    axs[0].set_title("Raw Signals vs. De-spiked Signals")
    axs[0].legend(); axs[0].grid(True)

    # 1: Highlighted substantial changes
    axs[1].plot(R_dh, label=f'|dR| ≥ {deriv_thresh}')
    axs[1].plot(G_dh, label=f'|dG| ≥ {deriv_thresh}')
    axs[1].plot(B_dh, label=f'|dB| ≥ {deriv_thresh}')
    axs[1].set_title("Highlighted Substantial Changes")
    axs[1].legend(); axs[1].grid(True)

    # 2: Absolute derivative + regions + peaks
    axs[2].plot(np.abs(R_dh), label='|dR|')
    axs[2].plot(np.abs(G_dh), label='|dG|')
    axs[2].plot(np.abs(B_dh), label='|dB|')
    axs[2].plot(merged_peaks, RGB_max_abs_d[merged_peaks], "rx", label="Merged Peaks")
    for x0, x1 in blank_ranges:
        axs[2].axvspan(x0, x1, color='lightgrey', alpha=0.4,
                       label='Blank Region' if x0 == blank_ranges[0][0] else "")
    for x0, x1 in pad_ranges:
        axs[2].axvspan(x0, x1, color='lightgreen', alpha=0.7,
                       label='Color Pad' if x0 == pad_ranges[0][0] else "")
    axs[2].set_title("Absolute Derivative + Region Classification")
    axs[2].legend(); axs[2].grid(True)

    # 3: The strip image
    axs[3].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axs[3].axis('off')
    axs[3].set_title("Cropped Strip Image")

    #plt.tight_layout()
    #plt.show()

def startUrineTestAnalysis(msg):
    ref_img_path, test_img_path, final_img_result_path, test_lib_path
    # Split: "OR_startUrineTestAnalysis~ref~test~final~lib"
    parts = msg.split("~")
    if len(parts) != 5 or parts[0] != "OR_startUrineTestAnalysis":
        print(f"[ERROR] Invalid urine msg format: {msg}")
        return None
        
    ref_img_path, test_img_path, final_img_result_path, test_lib_path = parts[1:5]
        
    # Validate paths (basic checks)
    if not all(path.strip() for path in parts[1:5]):
        print(f"[ERROR] Empty path in urine msg: {msg}")
        return None
        
    # Read test image
    test_img = cv2.imread(test_img_path)

    if test_img is None:
        print(f"Failed to read image at path: {test_img_path}")
        return None, None

    # Load test library JSON
    with open(test_lib_path, 'r') as file:
        data = json.load(file)

    # Deskew and crop
    test_processed_img, skew_angle = deskew_and_crop(test_img)

    # ROI extraction
    cropped_img = roiextraction(test_processed_img)

    # Optional quit check
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("Stopped by user")
        sys.exit()

    offset = 120
    test_img = cropped_img[:, offset:, :]

    # Cleaning intensity vs pixel graphs
    R_line, B_line, G_line, Y_line, R_nb, G_nb, B_nb, Y_nb = clean_signals(
        test_img,
        spike_thresh=16,
        max_spike_width=2
    )

    # Finding peaks
    merged_peaks, pad_ranges, blank_ranges = find_and_merge_peaks(
        R_nb, G_nb, B_nb, Y_nb,
        height_thresh=3,
        merge_distance=5
    )

    # Store pad regions
    pad_peak_pairs, merged_peaks_offset, pad_colours, all_crops = store_pad_regions(
        test_img,
        pad_ranges,
        offset=0,
        merged_peaks=None,
        output_dir='pad_crops'
    )

    # Convert RGB to LAB
    lab_colours = rgbtolab(pad_colours)

    # Analyte computations
    glucose = glucosecomp(lab_colours, data)
    bilirubin = bilirubincomp(lab_colours, data)
    ketones = ketonecomp(lab_colours, data)
    specificgravity = sgcomp(lab_colours, data)
    blood = bgcomp(lab_colours, data)
    pH = phcomp(lab_colours, data)
    protein = proteincomp(lab_colours, data)
    urobilinogen = urocomp(lab_colours, data)
    nitrites = nitritecomp(lab_colours, data)
    leukocytes = leukocytescomp(lab_colours, data)

    # Final image formation
    final_image = finalimageformation(all_crops)

    # Debug plotting
    plot_diagnostics(
        R_line, G_line, B_line,
        R_nb, G_nb, B_nb,
        merged_peaks,
        blank_ranges,
        pad_ranges,
        test_img
    )

    # Result dictionary
    result = {
        "Blood(BLO)": blood,
        "Bilirubin(BIL)": bilirubin,
        "Urobilinogen(URO)": urobilinogen,
        "Ketone(KET)": ketones,
        "Protein(PRO)": protein,
        "Nitrite(NIT)": nitrites,
        "Glucose(GLU)": glucose,
        "pH": pH,
        "Specific Gravity(SG)": specificgravity,
        "Leukocytes(LEU)": leukocytes
    }

    print(result)

    return result, final_image