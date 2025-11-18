import cv2
from skimage.draw import line
import numpy as np


class ROI:
    def __init__(self, image, radius=2):
        self.image = cv2.imread(image)
        print(self.image.shape)
        blank_space = np.all(self.image == [255, 255, 255], axis=2)
        self.mask = np.zeros(self.image.shape[:2], dtype=np.int16)
        self.mask[blank_space] = -1
        self.h, self.w = self.image.shape[:2]
        self.drawing = False
        self.previous = None
        self.radius = radius

    def add_point(self, x, y):
        for dx in range(-self.radius, self.radius):
            for dy in range(-self.radius, self.radius):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.w and 0 <= ny < self.h and self.mask[ny, nx] == 0:
                    self.mask[ny, nx] = 1
                    self.image[ny, nx] = [0, 0, 255]

    def add_points(self, points):
        for point in points:
            self.add_point(point[0], point[1])

    def click_event(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.previous = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing and (flags & cv2.EVENT_FLAG_LBUTTON):
            if self.previous is not None:
                rr, cc = line(self.previous[1], self.previous[0], y, x)
                new_pts = list(zip(cc, rr))
                self.add_points(new_pts)
                self.previous = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.drawing:
            self.drawing = False
            if self.previous is not None:
                rr, cc = line(self.previous[1], self.previous[0], y, x)
                new_pts = list(zip(cc, rr))
                self.add_points(new_pts)
            self.previous = None
        cv2.imshow('image', self.image)

    def show_image(self):
        cv2.imshow('image', self.image)
        cv2.setMouseCallback('image', self.click_event)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def get_roi(self):
        y, x = np.where(self.mask == 1)
        selected_roi = list(zip(x, y))
        return np.asarray(selected_roi)

    def get_mask(self):
        return (self.mask == 1).astype("float32")



