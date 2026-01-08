import cv2
import numpy as np
import mss
import pyautogui
import pydirectinput
import time
import keyboard
import threading

# Configuration
# =================================================================================================
# Colors to detect (HSV format). 
# Range for "Red" can be tricky because it wraps around 0/180 in HSV.
# Using two ranges for red is common.
LOWER_RED1 = np.array([0, 120, 70])
UPPER_RED1 = np.array([10, 255, 255])
LOWER_RED2 = np.array([170, 120, 70])
UPPER_RED2 = np.array([180, 255, 255])

# Detection Field of View (FOV)
# Only search in the center of the screen to increase performance and mimicking focus
FOV_SIZE = 400 

# Mouse settings
SMOOTHING = 3 # Higher = smoother/slower, Lower = snappier
TRIGGER_KEY = 'alt' # Key to hold to activate the aimbot
EXIT_KEY = 'q' # Key to exit the script
# =================================================================================================

# Screen resolution (Auto-detect)
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()
CENTER_X, CENTER_Y = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2

# Calculate the bounding box for our FOV
MONITOR = {
    "top": CENTER_Y - FOV_SIZE // 2,
    "left": CENTER_X - FOV_SIZE // 2,
    "width": FOV_SIZE,
    "height": FOV_SIZE
}

print(f"[*] Screen Resolution: {SCREEN_WIDTH}x{SCREEN_HEIGHT}")
print(f"[*] FOV Size: {FOV_SIZE}x{FOV_SIZE}")
print(f"[*] Hold '{TRIGGER_KEY}' to aim. Press '{EXIT_KEY}' to quit.")

sct = mss.mss()

def main():
    try:
        while True:
            if keyboard.is_pressed(EXIT_KEY):
                print("[*] Exiting...")
                break

            # Only process if trigger key is held
            if keyboard.is_pressed(TRIGGER_KEY):
                # 1. Capture the screen area (FOV)
                img = np.array(sct.grab(MONITOR))
                
                # Convert to HSV/BGR (mss returns BGRA)
                img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

                # 2. Color Detection (Masking)
                mask1 = cv2.inRange(hsv, LOWER_RED1, UPPER_RED1)
                mask2 = cv2.inRange(hsv, LOWER_RED2, UPPER_RED2)
                mask = mask1 + mask2 # Combine both red ranges

                # Clean up noise (optional, requires more CPU)
                # kernel = np.ones((3,3), np.uint8)
                # mask = cv2.dilate(mask, kernel, iterations=1)

                # 3. Find Contours
                contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                closest_target = None
                min_dist = float('inf')

                # Filter contours and find the closest to the center
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area > 50: # Minimum size to be considered a target
                        x, y, w, h = cv2.boundingRect(cnt)
                        
                        # Calculate center of the target relative to the FOV
                        target_x = x + w // 2
                        target_y = y + h // 2
                        
                        # Calculate distance from the exact center of the FOV (where the crosshair is)
                        fov_center = FOV_SIZE // 2
                        dist = ((target_x - fov_center) ** 2 + (target_y - fov_center) ** 2) ** 0.5

                        if dist < min_dist:
                            min_dist = dist
                            # Store target coordinates relative to screen
                            # MONITOR['left'] + target_x gives absolute screen matching X
                            closest_target = (MONITOR['left'] + target_x, MONITOR['top'] + target_y)

                # 4. Move Mouse
                if closest_target:
                    tx, ty = closest_target
                    
                    # Calculate relative movement needed
                    # pydirectinput works better for DirectX games
                    # We implement simple smoothing by dividing the distance
                    
                    offset_x = tx - CENTER_X
                    offset_y = ty - CENTER_Y
                    
                    # Move mouse
                    # Note: You might need to adjust sensitivity or division factor
                    if abs(offset_x) > 2 or abs(offset_y) > 2:
                        pydirectinput.move(int(offset_x // SMOOTHING), int(offset_y // SMOOTHING))
                        
                        # Optional: Auto-fire if very close to center
                        # if abs(offset_x) < 10 and abs(offset_y) < 10:
                        #     pydirectinput.click()

            # Optional: Show what the bot sees (Debug View)
            # CAUTION: This slows down the script significantly. Uncomment to test.
            cv2.imshow("Bot Vision", mask)
            cv2.waitKey(1)
            
            # Limited loop speed to save CPU
            time.sleep(0.01)

    except Exception as e:
        print(f"Error: {e}")
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
