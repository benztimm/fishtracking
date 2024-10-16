import numpy as np
import cv2
from scipy.fftpack import fft2, ifft2

# Helper function to create a Gaussian peak for correlation filtering
def create_gaussian_peak(size, sigma):
    w, h = size  # Now size contains (width, height) directly
    x, y = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
    d = np.sqrt(x * x + y * y)
    gaussian = np.exp(-d ** 2 / (2.0 * sigma ** 2))
    return gaussian

# Helper function for calculating correlation
def correlation_filtering(img, filter_kernel):
    img_fft = fft2(img)
    filter_fft = fft2(filter_kernel, img.shape)
    response = np.real(ifft2(img_fft * np.conjugate(filter_fft)))
    return response

# CSRT Tracker Class
class SimplifiedCSRT:
    def __init__(self, sigma=2.0, learning_rate=0.02):
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.template = None
        self.gaussian_peak = None
        self.filter_kernel = None
    
    def initialize(self, frame, bbox):
        x, y, w, h = bbox
        # Ensure the template size is (height, width) for image cropping
        self.template = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY).astype(np.float32)

        # Create a Gaussian peak with (width, height) from the bounding box
        self.gaussian_peak = create_gaussian_peak((w, h), self.sigma)

        # Perform FFT on the filter kernel and template
        self.filter_kernel = fft2(self.gaussian_peak) / (fft2(self.template) + 1e-5)
    
    def update(self, frame, bbox):
        x, y, w, h = bbox
        search_area = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY).astype(np.float32)
        response = correlation_filtering(search_area, self.filter_kernel)
        dy, dx = np.unravel_index(np.argmax(response), response.shape)
    
        # Extract the new template from the search area, ensuring we don't exceed the bounds
        dy = max(0, min(dy, search_area.shape[0] - self.template.shape[0]))
        dx = max(0, min(dx, search_area.shape[1] - self.template.shape[1]))
    
        new_template = search_area[dy:dy+self.template.shape[0], dx:dx+self.template.shape[1]]
    
        # Resize the new template to match the original template's size if necessary
        if new_template.shape != self.template.shape:
            new_template = cv2.resize(new_template, (self.template.shape[1], self.template.shape[0]))
    
        # Update the template and filter kernel (online learning)
        self.template = (1 - self.learning_rate) * self.template + self.learning_rate * new_template
        self.filter_kernel = (1 - self.learning_rate) * self.filter_kernel + self.learning_rate * fft2(self.gaussian_peak) / (fft2(self.template) + 1e-5)
    
        return int(x + dx), int(y + dy), w, h  # Updated bounding box

# Test with video
def test_tracker():
    cap = cv2.VideoCapture('video1.mp4')  # Replace with your video file

    # Initialize the tracker
    ret, frame = cap.read()
    bbox = cv2.selectROI(frame, False)
    tracker = SimplifiedCSRT()
    tracker.initialize(frame, bbox)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Update the tracker
        bbox = tracker.update(frame, bbox)

        # Draw the bounding box
        x, y, w, h = bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Display the result
        cv2.imshow('Tracking', frame)
        if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to quit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_tracker()
