import cv2
import numpy as np
import matplotlib # just added to prevent runtime problems
matplotlib.use('Agg') # just added to prevent runtime problems
import matplotlib.pyplot as plt
import os
from scipy.signal import savgol_filter

# --- Configuration ---

# Calibration: Define the vertical position of known volume marks as a percentage of the cylinder's height.
CALIBRATION_PERCENTAGES = [0.15, 0.85] # 500ml at 15% down, 100ml at 85% down.

# Analysis Parameters
ROI_DETECTION_DURATION_SEC = 5
DEBUG_VISUALIZATION = True

# Grace Period: Duration (in seconds) after ROI detection to allow free interface searching.
TRACKING_GRACE_PERIOD_SEC = 2.0

# --- NEW: Control the blur strength for interface detection ---
# A larger value (e.g., 5, 7) provides more smoothing but can reduce accuracy on sharp interfaces.
# A smaller value (e.g., 3) is a good compromise.
# Set to 1 to disable blurring completely. Must be an odd number.
INTERFACE_BLUR_KERNEL_SIZE = 3

# --- Helper Functions ---

def pixel_to_ml(y_pixel, calibration_points, total_volume=500, calib_volume_diff=400):
    """Converts a y-pixel coordinate within an ROI to a volume in ml."""
    pixel_500ml, pixel_100ml = calibration_points
    pixel_range = pixel_100ml - pixel_500ml
    if pixel_range <= 0: return 0
    relative_pos = (y_pixel - pixel_500ml) / pixel_range
    volume = total_volume - (relative_pos * calib_volume_diff)
    return np.clip(volume, 0, 500)


def find_cylinders_automatically(frame):
    """Automatically detects the two main cylinders in the frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidate_cylinders = []
    frame_height, frame_width = frame.shape[:2]

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = h / float(w) if w > 0 else 0
        area = cv2.contourArea(c)
        if 2.0 < aspect_ratio < 15 and h > frame_height * 0.4 and (frame_width * frame_height * 0.025) < area:
            candidate_cylinders.append((x, y, w, h))

    if len(candidate_cylinders) < 2: return None, None
    candidate_cylinders.sort(key=lambda b: b[2] * b[3], reverse=True)
    top_two = candidate_cylinders[:2]
    h1, h2 = top_two[0][3], top_two[1][3]
    if abs(h1 - h2) / max(h1, h2) > 0.25: return None, None
    top_two.sort(key=lambda b: b[0])
    return top_two[0], top_two[1]


def find_interface_level(cylinder_roi_img, calibration_points):
    """Finds the sediment-water interface using a brightness profile analysis."""
    if cylinder_roi_img is None or cylinder_roi_img.size == 0: return None, None

    gray = cv2.cvtColor(cylinder_roi_img, cv2.COLOR_BGR2GRAY)
    vertical_profile = np.mean(gray, axis=1)

    # --- MODIFIED: Use the tunable blur kernel from configuration ---
    if INTERFACE_BLUR_KERNEL_SIZE > 1:
        # Ensure kernel is odd
        kernel_size = INTERFACE_BLUR_KERNEL_SIZE if INTERFACE_BLUR_KERNEL_SIZE % 2 != 0 else INTERFACE_BLUR_KERNEL_SIZE + 1
        vertical_profile = cv2.GaussianBlur(vertical_profile, (1, kernel_size), 0).flatten()

    h = gray.shape[0]
    clear_zone_end = int(h * 0.1)
    sediment_zone_start = int(h * 0.9)
    if clear_zone_end <= 0 or sediment_zone_start >= h: return None, None

    clear_brightness = np.mean(vertical_profile[:clear_zone_end])
    sediment_brightness = np.mean(vertical_profile[sediment_zone_start:])
    if abs(clear_brightness - sediment_brightness) < 15: return None, None

    midpoint_brightness = (clear_brightness + sediment_brightness) / 2
    search_zone = vertical_profile[clear_zone_end:sediment_zone_start]
    abs_diff = np.abs(search_zone - midpoint_brightness)
    interface_y_relative = np.argmin(abs_diff) + clear_zone_end
    interface_level_ml = pixel_to_ml(interface_y_relative, calibration_points)
    return interface_level_ml, interface_y_relative


def analyze_video(video_path, start_time_sec, debug_filename, output_video_path):
    """Main function to process a video file with adaptive tracking logic."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): print(f"Error: Could not open video {video_path}"); return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    time_data, level_left_data, level_right_data = [], [], []
    rois_found = False
    roi_left, roi_right = None, None
    locked_level_left, locked_level_right = 500, 500
    current_y_left, current_y_right = None, None
    grace_period_end_time = float('inf')

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret: break

        current_time_sec = frame_count / fps

        if not rois_found and start_time_sec <= current_time_sec < (start_time_sec + ROI_DETECTION_DURATION_SEC):
            roi_left, roi_right = find_cylinders_automatically(frame)
            if roi_left is not None and roi_right is not None:
                rois_found = True
                grace_period_end_time = current_time_sec + TRACKING_GRACE_PERIOD_SEC
                print(f"[{video_path}] Cylinders found at {current_time_sec:.2f}s. Grace period ends at {grace_period_end_time:.2f}s.")

                _, _, _, lh = roi_left; calib_relative_left = [lh * p for p in CALIBRATION_PERCENTAGES]
                _, _, _, rh = roi_right; calib_relative_right = [rh * p for p in CALIBRATION_PERCENTAGES]

                if DEBUG_VISUALIZATION:
                    debug_frame = frame.copy()
                    x,y,w,h = roi_left; cv2.rectangle(debug_frame, (x,y), (x+w,y+h), (0, 255, 0), 3)
                    x,y,w,h = roi_right; cv2.rectangle(debug_frame, (x,y), (x+w,y+h), (0, 0, 255), 3)
                    cv2.imwrite(debug_filename, debug_frame)

        if not rois_found or current_time_sec < start_time_sec:
            frame_count += 1
            video_writer.write(frame)
            continue

        lx, ly, lw, lh = roi_left
        rx, ry, rw, rh = roi_right
        roi_left_img = frame[ly:ly+lh, lx:lx+lw]
        roi_right_img = frame[ry:ry+rh, rx:rx+rw]

        level_ml_left, y_left = find_interface_level(roi_left_img, calib_relative_left)
        level_ml_right, y_right = find_interface_level(roi_right_img, calib_relative_right)

        is_in_grace_period = current_time_sec < grace_period_end_time

        # --- Left Cylinder Logic ---
        if is_in_grace_period:
            if level_ml_left is not None:
                locked_level_left = level_ml_left; current_y_left = y_left
        else:
            if level_ml_left is not None and level_ml_left < locked_level_left:
                locked_level_left = level_ml_left; current_y_left = y_left

        # --- Right Cylinder Logic ---
        if is_in_grace_period:
            if level_ml_right is not None:
                locked_level_right = level_ml_right; current_y_right = y_right
        else:
            if level_ml_right is not None and level_ml_right < locked_level_right:
                locked_level_right = level_ml_right; current_y_right = y_right

        time_data.append(current_time_sec - start_time_sec)
        level_left_data.append(locked_level_left)
        level_right_data.append(locked_level_right)

        cv2.rectangle(frame, (lx, ly), (lx + lw, ly + lh), (0, 255, 0), 2)
        cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), (0, 0, 255), 2)

        if current_y_left is not None:
            cv2.line(frame, (lx, ly + current_y_left), (lx + lw, ly + current_y_left), (0, 255, 255), 2)
            cv2.putText(frame, f"{locked_level_left:.1f} ml", (lx + 5, ly + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if current_y_right is not None:
            cv2.line(frame, (rx, ry + current_y_right), (rx + rw, ry + current_y_right), (255, 0, 255), 2)
            cv2.putText(frame, f"{locked_level_right:.1f} ml", (rx + 5, ry + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

        video_writer.write(frame)
        frame_count += 1

    cap.release()
    video_writer.release()
    print(f"Certification video saved to {output_video_path}")
    if not rois_found: print(f"Error: Could not automatically detect two cylinders in {video_path}."); return None
    return time_data, level_left_data, level_right_data

def plot_results(time, level_left, level_right, title, left_label, right_label, output_filename):
    """Generates and saves a plot of the settling data."""
    plt.figure(figsize=(12, 7))
    if len(time) > 3:
        window_length = min(51, len(time) - (len(time) % 2 == 0))
        if window_length > 3:
            level_left_smooth = savgol_filter(level_left, window_length, 3)
            level_right_smooth = savgol_filter(level_right, window_length, 3)
        else:
            level_left_smooth, level_right_smooth = level_left, level_right
    else:
        level_left_smooth, level_right_smooth = level_left, level_right
    plt.plot(time, level_left, color='lightblue', alpha=0.4, label=f'{left_label} (Raw)')
    plt.plot(time, level_right, color='lightcoral', alpha=0.4, label=f'{right_label} (Raw)')
    plt.plot(time, level_left_smooth, label=f'{left_label} (Smoothed)', color='blue')
    plt.plot(time, level_right_smooth, label=f'{right_label} (Smoothed)', color='red')
    plt.title(title, fontsize=16); plt.xlabel("Time (seconds)", fontsize=12); plt.ylabel("Interface Level (ml)", fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5); plt.legend(); plt.gca().invert_yaxis()
    plt.ylim(510, 0); plt.xlim(left=0)
    plt.savefig(output_filename); plt.close()
    print(f"Plot saved to {output_filename}")


def run_analysis_on_video(video_path, start_time, left_label, right_label):
    """A wrapper function to run the full analysis for a single video."""
    if not os.path.exists(video_path): print(f"Error: Video file not found at {video_path}"); return
    base_filename = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = 'output'
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    output_graph_filename = os.path.join(output_dir, f"{base_filename}_graph.png")
    output_debug_filename = os.path.join(output_dir, f"{base_filename}_roi_detection.png")
    output_video_filename = os.path.join(output_dir, f"{base_filename}_analysis_video.mp4")
    print(f"--- Processing {video_path} ---")
    video_data = analyze_video(video_path, start_time, output_debug_filename, output_video_filename)
    if video_data:
        plot_results(video_data[0], video_data[1], video_data[2], f'Flocculant Performance: {os.path.basename(video_path)}', left_label, right_label, output_graph_filename)
        print(f"Analysis complete for {video_path}. Check the '{output_dir}' directory.\n")


# --- Main Execution Refactored for Django ---
def run_analysis_on_uploaded_video(video_path, start_time=1, left_label='Lab-Dried', right_label='Oven-Dried'):
    """
    Call this function after a video is uploaded to trigger the analysis.
    """
    run_analysis_on_video(video_path=video_path, start_time=start_time, left_label=left_label, right_label=right_label)