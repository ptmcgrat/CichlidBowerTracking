
import matplotlib.pyplot as plt
import matplotlib, datetime, cv2, pdb, os, sys, copy, warnings
import numpy as np

class PrepPreparer:

    def __init__(self, fileManager):
        self.__version__ = '1.0.0'
        self.fileManager = fileManager

    def downloadProjectData(self):
        self.fileManager.createDirectory(self.fileManager.localMasterDir)
        self.fileManager.createDirectory(self.fileManager.localAnalysisDir)
        self.fileManager.createDirectory(self.fileManager.localSummaryDir)
        self.fileManager.createDirectory(self.fileManager.localLogfileDir)
        self.fileManager.downloadData(self.fileManager.localPrepDir)
        self.fileManager.downloadData(self.fileManager.localLogfile)

    def validateInputData(self):
        assert os.path.exists(self.fileManager.localFirstFrame)
        assert os.path.exists(self.fileManager.localLastFrame)
        assert os.path.exists(self.fileManager.localPiRGB)
        assert os.path.exists(self.fileManager.localFirstDepthRGB)
        assert os.path.exists(self.fileManager.localLastDepthRGB)

        assert os.path.exists(self.fileManager.localSummaryDir)
        assert os.path.exists(self.fileManager.localAnalysisDir)

    def uploadProjectData(self, delete = True):
        self.fileManager.uploadData(self.fileManager.localDepthCropFile)
        self.fileManager.uploadData(self.fileManager.localTransMFile)
        self.fileManager.uploadData(self.fileManager.localVideoCropFile)
        self.fileManager.uploadData(self.fileManager.localPrepSummaryFigure)
        self.fileManager.uploadData(self.fileManager.localPrepLogfile)

        if delete:
            shutil.rmtree(self.localProjectDir)

    def prepData(self):
        self._createLogFile()
        self._cropDepth()
        self._cropAndRegisterVideo()
        self._summarizePrep()

    def _createLogFile(self):
        with open(self.fileManager.localPrepLogfile,'w') as f:
            print('PythonVersion: ' + sys.version.replace('\n', ' '), file = f)
            print('NumpyVersion: ' + np.__version__, file = f)
            print('MatplotlibVersion: ' + matplotlib.__version__, file = f)
            print('OpenCVVersion: ' + cv2.__version__, file = f)
            print('Username: ' + os.getenv('USER'), file = f)
            print('Nodename: ' + os.uname().nodename, file = f)
            print('DateAnalyzed: ' + str(datetime.datetime.now()), file = f)

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

    def _cropDepth(self):

        # Depth data is assumed to be in meters

        # Read in depth and RGB data
        firstFrame = np.load(self.fileManager.localFirstFrame)
        lastFrame = np.load(self.fileManager.localLastFrame)
        lastDepthRGB = cv2.imread(self.fileManager.localLastDepthRGB)
        firstDepthRGB = cv2.imread(self.fileManager.localFirstDepthRGB)

        # Calculate depth change and average data for removing extreme data. 
        difference = lastFrame - firstFrame
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            average_depth = np.nanmean([lastFrame,firstFrame], axis = 0) # Calculate average height to help exclude things that are consistently out of frame

        median_height = np.nanmedian(firstFrame) # Calculate average height of sand
        difference[(average_depth > median_height + 4) | (average_depth < median_height - 8)] = np.nan # Filter out data 4cm lower and 8cm higher than tray

        # Create color map for depth data
        cmap = copy.copy(matplotlib.cm.get_cmap("jet"))
#       cmap = plt.get_cmap('jet')
        cmap.set_bad(color = 'black')

        # Create images for display
        depth_change_cmap = cmap(plt.Normalize(-5,5)(difference)) # Normalize data that is +- 5 cm
        first_depth_cmap = cmap(plt.Normalize(median_height - 5, median_height + 5)(firstFrame))
        last_depth_cmap = cmap(plt.Normalize(median_height - 5, median_height + 5)(lastFrame))
        
        # Loop until an acceptable box is created
        while True:
            # Query user to identify regions of the tray that are good
            self.poly = []
            self.original_pic = lastDepthRGB
            self.interactive_pic = self.original_pic.copy()
            self.interactive_text = 'Click four points to crop. Right-click to start over. Press escape once you are finished'

            cv2.imshow(self.interactive_text, self.interactive_pic)
            cv2.setMouseCallback(self.interactive_text, self._click_event)
            cv2.waitKey(0)

            for i in range(3):
                cv2.destroyAllWindows()
                cv2.waitKey(1)

            if len(self.poly) != 4:
                continue

            depth_polys = self.poly

            # Create figure for user to see the crop
            fig = plt.figure(figsize=(9, 9))
            ax1 = fig.add_subplot(2,2,1)       
            ax2 = fig.add_subplot(2,2,2)
            ax3 = fig.add_subplot(2,2,3)
            ax4 = fig.add_subplot(2,2,4)
            ax1.imshow(lastDepthRGB)
            ax1.add_patch(matplotlib.patches.Polygon(depth_polys, color="orange", fill = False, lw = 3.0))
            ax1.set_title("Last Depth RGB")
            ax2.imshow(depth_change_cmap)
            ax2.add_patch(matplotlib.patches.Polygon(depth_polys, color="orange", fill = False, lw = 3.0))
            ax2.set_title("Depth change over whole trial")
            ax3.imshow(firstDepthRGB)
            ax3.add_patch(matplotlib.patches.Polygon(depth_polys, color="orange", fill = False, lw = 3.0))
            ax3.set_title("First DepthRGB")
            ax4.imshow(last_depth_cmap)
            ax4.add_patch(matplotlib.patches.Polygon(depth_polys, color="orange", fill = False, lw = 3.0))
            ax4.set_title("Depth at late time point")
            fig.canvas.manager.set_window_title('Close window and type q in terminal if this is acceptable')
            plt.show()

            userInput = input('Type q if this is acceptable: ')
            if userInput == 'q':
                break

        # Save and back up tray file
        self.labeledDepthRGB = self.interactive_pic.copy()
        self.labeledDepthPoints = self.poly
        with open(self.fileManager.localDepthCropFile, 'w') as f:
            print(','.join([str(x) for x in self.poly]), file = f)

    def _cropAndRegisterVideo(self):
        im1 =  cv2.imread(self.fileManager.localPiRGB)
        resized_depth_image = cv2.resize(self.labeledDepthRGB, im1.shape[0:2][::-1])
        depthRGB = cv2.imread(self.fileManager.localFirstDepthRGB)

        while True:
            self.poly = []
            self.original_pic = np.hstack([im1,resized_depth_image])
            self.interactive_pic = self.original_pic.copy()
            self.interactive_text = 'Click the same four points on the Raspberry Pi image. Press escape once you are finished'

            cv2.imshow(self.interactive_text, self.interactive_pic)
            cv2.setMouseCallback(self.interactive_text, self._click_event)
            cv2.waitKey(0)

            for i in range(3):
                cv2.destroyAllWindows()
                cv2.waitKey(1)
        
            self.transM = cv2.getPerspectiveTransform(np.float32(self.poly),np.float32(self.labeledDepthPoints))
            newImage = cv2.warpPerspective(im1, self.transM, (640, 480))

            fig = plt.figure(figsize=(18, 12))
            ax1 = fig.add_subplot(1,2,1)       
            ax2 = fig.add_subplot(1,2,2)
        
            ax1.imshow(depthRGB)
            ax1.set_title("Depth RGB image")

            ax2.imshow(newImage)
            ax2.set_title("Registered Pi RGB image")

            #fig.savefig(self.localMasterDirectory + self.transFig)
            fig.canvas.manager.set_window_title('Close window and type q in terminal if this is acceptable')
            plt.show()

            userInput = input('Type q if this is acceptable: ')
            if userInput == 'q':
                break

        self.labeledVideoPoints = self.poly

        with open(self.fileManager.localVideoCropFile, 'w') as f:
            print(','.join([str(x) for x in self.poly]), file = f)
        np.save(self.fileManager.localTransMFile, self.transM)


    def _summarizePrep(self):

        firstFrame = np.load(self.fileManager.localFirstFrame)
        lastFrame = np.load(self.fileManager.localLastFrame)
        depthRGB = cv2.imread(self.fileManager.localFirstDepthRGB)
        #depthRGB = cv2.cvtColor(depthRGB,cv2.COLOR_BGR2GRAY)
        piRGB =  cv2.imread(self.fileManager.localPiRGB)

        cmap = copy.copy(matplotlib.cm.get_cmap("jet"))
        cmap.set_bad(color = 'black')

        fig = plt.figure(figsize=(12, 12))
        ax1 = fig.add_subplot(2,2,1)       
        ax2 = fig.add_subplot(2,2,2)
        ax3 = fig.add_subplot(2,2,3)
        ax4 = fig.add_subplot(2,2,4)

        ax1.imshow(depthRGB, cmap = 'gray')
        ax1.add_patch(matplotlib.patches.Polygon(self.labeledDepthPoints, color="orange", fill = False, lw = 3.0))
        ax1.set_title("Depth RGB image with depth crop")

        ax2.imshow(lastFrame - firstFrame, cmap = cmap)
        ax2.add_patch(matplotlib.patches.Polygon(self.labeledDepthPoints, color="orange", fill = False, lw = 3.0))
        ax2.set_title("Total trial depth change image with depth crop")
    
        ax3.imshow(piRGB, cmap='gray')
        ax3.add_patch(matplotlib.patches.Polygon(self.labeledVideoPoints, color="orange", fill = False, lw = 3.0))
        ax3.set_title("Pi RGB image with video crop")

        warpedPiRGB = cv2.warpPerspective(piRGB, self.transM, (640, 480))
        ax4.imshow(warpedPiRGB, cmap = 'gray')
        ax4.add_patch(matplotlib.patches.Polygon(self.labeledDepthPoints, color="orange", fill = False, lw = 3.0))
        ax4.set_title("Registered Pi RGB image with video and depth crop")

        fig.savefig(self.fileManager.localPrepSummaryFigure, dpi=300)

        #plt.show()

