import sys
import cv2
from rdp import rdp
import matplotlib.pyplot as plt

sys.setrecursionlimit(1000000000)

class MaskToPolygon:
    
    # global area
    points = []
    dx = [0, 1, 1, 1, 0, -1, -1, -1]
    dy = [-1, -1, 0, 1, 1, 1, 0, -1]
    
    def __init__(self, epsilon:float = 5., threshold:int = 100):
        self.epsilon = epsilon
        self.threshold = threshold
        
    def __call__(self, image: np.ndarray, option:str="B"):
        image = image.astype(np.uint8) * 255
        image = self.image_dilation_erosion(image)
        
        if image.shape[0] != 512 or image.shape[1] != 512:
            raise Exception("Invalid Shape Error\n\n image.shape must (512, 512, 3) and must use 3 channel")
        
        if option == "B":
            crop_image = image[...,0]
            vinyl_image = image[...,1]
            crop_mask = self._call_option_single(crop_image)
            vinyl_mask = self._call_option_single(vinyl_image)
            remain_mask = np.clip(np.ones((512, 512))*255 - crop_mask - vinyl_mask, 0, 255)
            res_mask = np.concatenate((crop_mask[..., np.newaxis]
                                       , vinyl_mask[..., np.newaxis]
                                       , remain_mask[..., np.newaxis]), -1)
            return res_mask / 255.
        elif option == "C":
            image = image[...,0]
            return np.array(self._call_option_single(image))/255.
        elif option == "V":
            image = image[...,1]
            return np.array(self._call_option_single(image))/255.
        else:
            raise Exception("B(BOTH), C(CROP), V(VINYL) 중 올바른 option 값을 주십시오. \n 기본값은 C(CROP)입니다.")
            
        
    def _call_option_single(self, org_image):
        self.points = []
        contour_img = self.mask_to_contour(org_image)
        self.contour_img = np.array(contour_img)
        # self._visualize(org_image, self.contour_img)
        
        self._get_points()
        # cleared_contour  = self._points_to_contour(self.points)
        # cleared_mask     = self._points_to_mask(self.points)
        # self._visualize(cleared_mask, cleared_contour)
        
        rdp_points       = self._get_rdp_point()
        rdp_mask         = self._points_to_mask(rdp_points)
        # rdp_contour      = self.mask_to_contour(rdp_mask)
        # self._visualize(rdp_mask, rdp_contour)
        
        return rdp_mask
    
    def image_dilation_erosion(self, image):
        kernel = np.ones((3, 3), np.uint8)
        dilation = cv2.dilate(image, kernel, iterations=2)
        erosion = cv2.erode(dilation, kernel, iterations=1)
        return erosion
    
    def _get_rdp_point(self):
        rdp_points = []
        for section_points in self.points:
            section_points = np.array(section_points)
            simplified_points = rdp(section_points, epsilon=self.epsilon)
            rdp_points.append(list(simplified_points))
        # rdp_points = map(list, rdp_points)
        return rdp_points
    
    def _points_to_mask(self, points):
        mask = np.zeros_like(self.contour_img, dtype=np.uint8)
        for section_points in points:
            cv2.fillPoly(mask, np.expand_dims(section_points, 0), color=(255))
        return mask
    
    def _points_to_contour(self, points):
        contour_img = np.zeros_like(self.contour_img)
        if not points:
            return contour_img
        cv2.drawContours(contour_img, np.expand_dims(np.array(sum(points, [])), -2), -1, (255, 255, 255), 1)
        return contour_img
    
    def _dfs(self, x, y, segment):
        self.contour_img[y][x] = 0
        for i, j in zip(self.dx, self.dy):
            nx = x + i
            ny = y + j
            if 0 <= nx < 512 and 0 <= ny < 512 and self.contour_img[ny][nx]:
                self.points[segment].append([nx, ny])
                self._dfs(nx, ny, segment)
    
    def _get_points(self, segment=0):
        while np.sum(self.contour_img):
            image_max = tf.reshape(self.contour_img, -1)
            image_max_idx = tf.argmax(image_max)
            y = int(image_max_idx) // 512
            x = int(image_max_idx) % 512
            self.points.append([[x, y]])
            self._dfs(x, y, segment)
            if len(self.points[segment]) < self.threshold:
                self.points.pop()
                segment -= 1
            segment += 1
            
    
    def mask_to_contour(self, image):
        contours, _ = cv2.findContours(image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        contour_img = np.zeros_like(image)
        cv2.drawContours(contour_img, contours, -1, (255, 255, 255), 1)
        return contour_img
    
    def _visualize(self, img, contour_img = None):
        if contour_img is None:
            plt.imshow(img)
            plt.show()
        else:
            plt.figure(figsize=(10, 10))
            plt.subplot(1, 2, 1)
            plt.imshow(img)
            plt.subplot(1, 2, 2)
            plt.imshow(contour_img)


import time
start = time.time()
MTP = MaskToPolygon()
image = np.array(pred[13])