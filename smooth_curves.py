import cv2
import numpy as np

def coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1 * (3 / 5))

    # Check if the slope is too small (close to zero)
    if abs(slope) > 1e-5:
        # Calculate x-coordinates using the slope and intercept
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
    else:
        # Default x-coordinates if the slope is too small
        x1 = x2 = image.shape[1] // 2

    return np.array([x1, y1, x2, y2])

def detect_headlights(image):
    # Convert the image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define a range for the color of the car headlights
    #lower_headlight = np.array([0, 0, 200], dtype=np.uint8)
    #upper_headlight = np.array([180, 30, 255], dtype=np.uint8)
    lower_red = np.array([0, 100, 100], dtype=np.uint8)
    upper_red = np.array([10, 255, 255], dtype=np.uint8)
    
    # Create a mask based on the color range
    mask = cv2.inRange(hsv, lower_red, upper_red)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw rectangles around the detected headlights
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    return mask

#handle turns in the road 
def smooth_lines(current, previous, alpha=0.2):
    if current is None:
        return previous
    elif previous is None:
        return current
    elif len(current) == 0 or len(previous) == 0:
        return current if len(current) > 0 else previous
    else:
        min_length = min(len(current), len(previous))
        current = current[:min_length]
        previous = previous[:min_length]

        # Ensure both current and previous are not empty
        if len(current) > 0 and len(previous) > 0:
            # Smooth the lines using exponential moving average
            smoothed = alpha * previous + (1 - alpha) * current
            return smoothed
        else:
            return current if len(current) > 0 else previous



def fit_curve(points):
    try:
        # Fit a curve to the points using polynomial regression
        curve_fit = np.polyfit(points[:, 1], points[:, 0], 2)
        # Generate a polynomial from the coefficients
        curve = np.poly1d(curve_fit)
        return curve
    except np.RankWarning:
        print("Polyfit warning: The matrix may be poorly conditioned. Skipping curve fitting.")
        return None



def average_slope(image, lines, curve_fit_threshold=0.5):
    left_fit = []  
    right_fit = []  

    if lines is not None and len(lines) > 0:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            intercept = parameters[1]

            # Check if the slope is on the left or right
            # left -> negative, right -> positive
            if slope < 0:
                left_fit.append((slope, intercept))
            else:
                # Fit a curve only when the slope is above the threshold
                if abs(slope) > curve_fit_threshold:
                    right_fit.append(fit_curve(np.array([(x1, y1), (x2, y2)])))

        # Check if there are lines on both sides
        if len(left_fit) > 0 and len(right_fit) > 0:
            # Average into a single slope and intercept
            left_fit_avg = np.average(left_fit, axis=0)
            right_fit_avg = np.average(right_fit, axis=0)

            # Calculate coordinates for left and right lines
            left_line = coordinates(image, left_fit_avg)
            right_line = fit_curve_coordinates(image, right_fit_avg)

            return left_line, right_line

    # Return empty arrays if no lines or curves are detected
    return np.array([]), np.array([])

def fit_curve_coordinates(image, curve_fit):
    if curve_fit is not None and callable(curve_fit):
        # Generate x values for the curve
        x_values = np.linspace(0, image.shape[0], num=100)
        # Calculate corresponding y values using the curve
        y_values = curve_fit(x_values).astype(np.int32)

        # Convert the coordinates to integer
        curve_points = np.column_stack((y_values, x_values)).astype(np.int32)
        return curve_points
    else:
        return np.array([])  # Return an empty array if the curve_fit is not callable or None




def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny

def display(image, curves, bounding_box):
    line_image = np.zeros_like(image)

    x1, y1, x2, y2 = bounding_box
    cv2.rectangle(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    mask = np.zeros_like(line_image)
    cv2.fillPoly(mask, np.array([[(x1, y1), (x2, y1), (x2, y2), (x1, y2)]], dtype=np.int32), 255)
    line_image = cv2.bitwise_and(line_image, mask)

    if curves is not None and len(curves) == 2:
        for curve in curves:
            if curve.size > 0:
                curve_points = curve.reshape(-1, 1, 2)
                cv2.polylines(line_image, [curve_points], isClosed=False, color=(0, 0, 255), thickness=10)

    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return line_image




def region(image):
    height = image.shape[0]
    polygons = np.array([[(200, height), (1100, height), (550, 250)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


def preprocess_frame(frame):
    # Resize the frame
    processed_frame = cv2.resize(frame, (1164, 874))
    #processed_frame = color_filter(processed_frame)
    return processed_frame

#dont use for now 
def color_filter(image):
    lower_yellow = np.array([0, 100, 100], dtype=np.uint8)
    upper_yellow = np.array([30, 255, 255], dtype=np.uint8)
    lower_white = np.array([180, 180, 180], dtype=np.uint8)
    upper_white = np.array([255, 255, 255], dtype=np.uint8)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_white = cv2.inRange(image, lower_white, upper_white)

    combined_mask = cv2.bitwise_or(mask_yellow, mask_white)

    filtered_image = cv2.bitwise_and(image, image, mask=combined_mask)

    return filtered_image

def calculate_pitch_and_yaw(image, averaged_lines):
    left_line, right_line = averaged_lines

    # Check if both left_line and right_line are not None
    if left_line is not None and right_line is not None:
        # Extract coordinates from left and right lines or curves
        x1_left, y1_left, x2_left, y2_left = extract_coordinates(left_line, image.shape[0])
        x1_right, y1_right, x2_right, y2_right = extract_coordinates(right_line, image.shape[0])

        # Calculate vanishing point (intersection of left and right lines)
        try:
            vanishing_point = np.linalg.solve(
                [[y1_left - y2_left, x1_left - x2_left],
                 [y1_right - y2_right, x1_right - x2_right]],
                [x1_left * (y1_left - y2_left) - y1_left * (x1_left - x2_left),
                 x1_right * (y1_right - y2_right) - y1_right * (x1_right - x2_right)]
            )
        except np.linalg.LinAlgError:
            print("Singular matrix encountered. Unable to calculate vanishing point.")
            return None

        # Calculate horizon line (middle of the image)
        horizon_line = [image.shape[1] // 2, 0, image.shape[1] // 2, image.shape[0]]

        # Calculate pitch angle
        pitch_angle = np.arctan2(vanishing_point[1] - horizon_line[1], vanishing_point[0] - horizon_line[0])

        # Calculate yaw angle
        yaw_angle = np.arctan2(vanishing_point[1] - image.shape[0] // 2, vanishing_point[0] - image.shape[1] // 2)

        return np.degrees(pitch_angle), np.degrees(yaw_angle)

    # Return None if either left_line or right_line is None
    return None

def extract_coordinates(line_or_curve, height):
    if len(line_or_curve) > 0:
        if isinstance(line_or_curve[0], np.poly1d):
            # If it's a curve (poly1d), evaluate it at the bottom and top of the image
            x_bottom = int(line_or_curve[0](height))
            x_top = int(line_or_curve[0](0))
            return x_bottom, height, x_top, 0
        else:
            # If it's a line, use the coordinates directly
            return line_or_curve
    else:
        return (0, 0, 0, 0)  # Return a dummy line if line_or_curve is empty







# Video path
#test already labeled points
video_path = '/home/cropthecoder/Documents/Comma_AI/calib_challenge/labeled/0.hevc'

angles_list = []

# Open the video file
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
video_duration = total_frames / fps

desired_values = 1200
previous_left_curve = None
previous_right_curve = None



# Loop through each frame of the video
while True:
    # Read a frame from the video
    ret, frame = cap.read()

    # Break the loop if the video is finished
    if not ret:
        break

    # Perform preprocessing on the frame
    processed_frame = preprocess_frame(frame)

    headlights_mask = detect_headlights(processed_frame)
    

    # Perform lane detection on the preprocessed frame
    canny_image = canny(processed_frame)
    cropped_image = region(canny_image)
    cropped_image_no_headlights = cv2.bitwise_and(cropped_image, cropped_image, mask=cv2.bitwise_not(headlights_mask))

    lines = cv2.HoughLinesP(cropped_image_no_headlights, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    left_curve, right_curve = average_slope(processed_frame, lines)
     # Smooth the left and right curves
    left_curve = smooth_lines(left_curve, previous_left_curve)
    right_curve = smooth_lines(right_curve, previous_right_curve)

    # Store the current curves for the next iteration
    previous_left_curve = left_curve
    previous_right_curve = right_curve

    # Smooth the left and right curves
    averaged_lines = average_slope(processed_frame, lines)

    # Check if any lines are detected before calculating pitch and yaw
    if len(averaged_lines) > 0:
        bounding_box = (450, 720, 740, 650)
        line_image = display(processed_frame, averaged_lines, bounding_box)
        combo = cv2.addWeighted(processed_frame, 0.8, line_image, 1, 1)

        # Display the original and processed frames
        cv2.imshow('Original Frame', frame)
        cv2.imshow('Processed Frame', combo)

        # Calculate pitch and yaw
        result = calculate_pitch_and_yaw(processed_frame, averaged_lines)
        if result is not None:
            pitch, yaw = result
            #print(f"Pitch: {pitch} degrees, Yaw: {yaw} degrees")
            #angles_list.append([pitch, yaw])
            pitch_rad = np.radians(pitch)
            yaw_rad = np.radians(yaw)
            print(f"Pitch: {pitch_rad} radians, Yaw: {yaw_rad} radians")
            angles_list.append([pitch_rad, yaw_rad])
    
            
   
    # Break the loop if 'q' key is pressed
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

angles_array = np.array(angles_list)

# Print the resulting array
print("Output Data (Pitch, Yaw):")
print(angles_array)

np.savetxt('0.txt', angles_array, fmt='%.6f', delimiter=' ', header='', comments='')


# Release resources
cap.release()
# Uncomment the following line if you want to save the processed video
# out.release()
cv2.destroyAllWindows()
