from helper_modules.file_manager import FileManager as FM

import numpy as np
import os, sys, pdb, subprocess, argparse
import cv2
import matplotlib

class Crop():
    
    def __init__(self, img):
          self.img = img

    def _click_event(self, event, x, y, flags, params):
        
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.poly) == 4:
                return
            self.poly.append((x,y))
            cv2.circle(self.interactive_pic, (x,y), 5, (255,0,0), -1)
            cv2.putText(self.interactive_pic, str(len(self.poly)), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,0), 2)
            if len(self.poly) > 1:
                for i in range(len(self.poly) - 1):
                    cv2.line(self.interactive_pic, self.poly[i], self.poly[i+1], (255,0,0), 1)
            cv2.imshow(self.interactive_text, self.interactive_pic)

        elif event == cv2.EVENT_RBUTTONDOWN:
            self.poly = []
            self.interactive_pic = self.original_pic.copy()
            cv2.imshow(self.interactive_text, self.interactive_pic)

    def createCropFile(self,fm_obj, tankID):

        self.poly = []
        self.original_pic = self.img
        self.interactive_pic = self.original_pic.copy()
        self.interactive_text = 'Click four points to crop. Right-click to start over. Press escape once you are finished'

        cv2.imshow(self.interactive_text, self.interactive_pic)
        cv2.setMouseCallback(self.interactive_text, self._click_event)
        cv2.waitKey(0)

        for i in range(3):
            cv2.destroyAllWindows()
            cv2.waitKey(1)

        assert len(self.poly) == 4
        # if len(self.poly) != 4:
        #     continue
        
                    
        depthCropFile = fm_obj.localMasterDir + '__TankData/' + tankID + '/DepthCrop.txt'

        os.makedirs(os.path.dirname(depthCropFile),exist_ok=True)

        with open(depthCropFile, 'w') as f:
            print(','.join([str(x) for x in self.poly]), file = f)
                        
        fm_obj.uploadData(depthCropFile)

        mask = np.zeros_like(img, dtype=np.uint8)
        points = np.array(self.poly, dtype=np.int32).reshape((-1, 1, 2))
        # points = np.array(self.poly)
        # mask = np.zeros_like(image, dtype=np.uint8)
        cv2.fillPoly(mask, [points], (255, 255, 255))
        cropped_image = cv2.bitwise_and(self.img, mask)

        croppedImg = os.path.dirname(depthCropFile) + '/MaskedImg.jpg'
        cv2.imwrite(croppedImg, cropped_image)

        fm_obj.uploadData(croppedImg)

parser = argparse.ArgumentParser()
parser.add_argument('tankID', type = str, help = 'Tank ID')

args = parser.parse_args()

fm_obj = FM()
# prep_preparer = PrP(fileManager=file_manager)
tankID = args.tankID

raw_data = fm_obj.localMasterDir + "__TankData/"+ tankID + "/FirstDepthRGB.jpg"
assert fm_obj.checkFileExists(raw_data)
fm_obj.downloadData(raw_data)
img = cv2.imread(raw_data)
crop_img = Crop(img)
crop_img.createCropFile(fm_obj,tankID)
