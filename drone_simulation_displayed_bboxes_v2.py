from direct.showbase.ShowBase import ShowBase
from direct.actor.Actor import Actor
from panda3d.core import AmbientLight, DirectionalLight, Vec4, loadPrcFileData, GraphicsOutput, PointLight, Spotlight, PerspectiveLens, Camera, TextNode, PNMImage, Filename, WindowProperties, NodePath, LineSegs, TransparencyAttrib
from direct.gui.OnscreenText import OnscreenText
from direct.gui.DirectGui import DirectFrame, OnscreenText
from direct.task import Task
#from wand.image import Image
import math, cv2, random, os #anche se non illumina cv2, il pacchetto è installato e funziona
import numpy as np
import subprocess
from ultralytics import YOLO

'''
Algorithm steps:
Step 1: Capture two images using the stereo camera setup.
Step 2: Use the forward projection to map world points to image points using the formula x=f*X/Z and y=f*Y/Z.
Step 3: Utilize the homogeneous coordinates to find the 2D point based on the coordinates (x, y, z).
Step 4: Apply the projection matrix Mprojection and other matrices like Mextrinsics and Maffine to handle the camera's intrinsic and extrinsic parameters.
Step 5: Address lens distortions, both radial and tangential, to correct the image.
Step 6: Implement the stereo rectification process to make the left and right image planes parallel. This simplifies the stereo matching process.
Step 7: Create a disparity map using the rectified images. The disparity d is calculated as the difference between the x-coordinates of the projected points on the left and right image planes.
Step 8: Calculate the distance Z of the object from the camera using the formula Z=(f*B)/d, where f is the focal length, B is the distance between the camera centers, and d is the disparity.'''

def predict(model, filename):

    results = model(filename, conf=0.2)  #il treshold di confidence è a discrezione

    for result in results:
        boxes = result.boxes

        tpclass = boxes.cls.detach().cpu().numpy() #sono tensori GPU, vanno staccati

        if not (tpclass.size == 0): #se non ci sono classificazioni, l'etichetta è un tensore vuoto
            
            result.save(filename='prediction'+filename)  #salva l'immagine con le bounding box
            return(True,boxes.xywh.detach().cpu().numpy())

        else:
            return(False,False)

def pnmimage_to_cv2mat(pnm_image):
    x_size = pnm_image.get_x_size()
    y_size = pnm_image.get_y_size()
    img_array = np.zeros((y_size, x_size, 3), dtype=np.uint8)

    for x in range(x_size):
        for y in range(y_size):
            r, g, b, a = pnm_image.get_pixel(x, y)
            img_array[y, x] = (r, g, b)

    return img_array

def pixels_to_dframe(p_x, p_y, p_w, p_h): #funzione per il mapping da coordinate bounding box YOLO a coordinate di Panda3D che nello schermo vanno da -1 a 1
    n_x = (p_x / (1024 / 2)) - 1
    n_y = 1-(p_y / (512))
    n_w = p_w / (1024 / 2)
    n_h = p_h / (512)
    return n_x, n_y, n_w, n_h

def yolo_bb_convert(bb_str): #funzione per covertire la stringa printata da yolo_detection.py in una lista usabile
    k = bb_str.replace("[","").replace("]","").replace(" ","")
    k = list(k.split(",,\n"))
    for y in range(len(k)):
        k[y] = list(k[y].split(",,"))

    out = [[round(float(number)) for number in row] for row in k] #per evitare errori arrotondiamo i valori dei pixel a numeri interi

    return out

def get_intrinsic_matrix(lens, imageh, imagew):
    hfov, vfov = lens.getFov()
    aspect_ratio = lens.getAspectRatio()
    
    #calcola le lunghezze focali
    fx = (imagew / 2) / np.tan(np.radians(hfov / 2))
    fy = (imageh / 2) / np.tan(np.radians(vfov / 2))

    cx = imagew/2
    cy = imageh/2

    K = np.array([ #K è la moltiplicazione di M_affine e M_projection
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    
    return K,fx

#funzione per aggiungere radial distortion (non necessaria)
'''def add_distortion_example(img1):
    with Image(filename=img1) as img:
        img.virtual_pixel = 'transparent'
        img.distort('barrel', (0.2, 0.0, 0.0, 1.0))
        img.save(filename='distorted_image1.png')'''


#calcolo della distanza usando la funzione Z=f*B/d su diversi punti + mean absolute deviation
def calculate_disparity_and_distance(rectified_img_1,rectified_img_2,bboxes,w1,h1,baseline,fx,rendered): #le boundingboxes sono array (x,y,width,height) che contengono gli elementi trovati

    if rendered >= 3: #necessario perché boh sennò non funziona essendo il modello caricato in async

        distances = []

        for bbox in bboxes:

            #per ogni bounding box calcoliamo la disparità su un numero arbitrario di pixel nell'area

            z_values = np.empty(0) #distanze calcolate
            density = math.ceil(min(bbox[2],bbox[3])/4) #quante volte viene effettuato il window matching con sliding window


            for p_w in range(math.floor(bbox[2]/density)): 
                for p_h in range(math.floor(bbox[3]/density)):

                    add_w, add_h = p_w*density, p_h*density

                    if add_w + bbox[0] + bbox[2] > w1:
                        #se questa condizione è soddisfatta, spostare a destra la bounding box la fa uscire dallo schermo, quindi la facciamo andare dall'altra parte se sta per uscire
                        add_w = w1 - add_w - bbox[0] - bbox[2]

                    if add_h + bbox[1] + bbox[3] > h1:
                        #se questa condizione è soddisfatta, spostare in giù la bounding box la fa uscire dallo schermo, quindi la facciamo andare dall'altra parte se sta per uscire
                        add_h = h1 - add_h - bbox[1] - bbox[3]

                    cropped_image = rectified_img_1[bbox[1]+add_h:bbox[1]+bbox[3]+add_h,bbox[0]+add_w:bbox[0]+bbox[2]+add_w] #questo tipo di cropping cursed è possibile con opencv (https://stackoverflow.com/questions/15589517/how-to-crop-an-image-in-opencv-using-python)
                    matched_image = cv2.matchTemplate(rectified_img_2, cropped_image, cv2.TM_CCORR_NORMED) #applica normalized cross-correlation con cv2

                    #trova le migliori corrispondenze per il pixel bbox[1];bbox[]
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(matched_image)
                    match_x, match_y = max_loc

                    #calcola disparità tra i pixel
                    disparity = abs(bbox[0]+add_w - match_x)
                    disparity_y = abs(bbox[1]+add_h - match_y)
                    #print("disparities: ",disparity,disparity_y)

                    #calcolo distanza del punto dalla camera
                    Z = (fx * baseline) / disparity if disparity != 0 else 0
                    z_values = np.append(z_values,Z)

            #dato che abbiamo preso dei punti a caso nella bounding box, potrebbero esserci dei punti a un'altezza più elevata (es. un pezzettino di tetto) o errori
            #per rimuovere gli outliers i dati (le distanze) sono monodimensionali, quindi niente RANSAC. Basta una Mean Absolute Deviation (MAD)
                    
            #print("z values: ",z_values)
            median = np.median(z_values)
            absolute_deviation = np.median(np.abs(z_values - median)) #mediana degli scarti
            it = np.nditer(z_values, flags=['f_index'])

            for i in it:
                if i < median - 3*absolute_deviation or i > median + 3*absolute_deviation: #3 è il threshold
                    np.delete(z_values,it.index)

            #una volta rimossi gli outlier calcoliamo la mediana delle distanze, che dovrebbe dare un'indicazione approssimativamente fedele
            likely_distance = np.median(z_values)
            distances.append(likely_distance)

        return distances
    else:
        return [0]
        



#attiva filtro antialiasing 4x MSAA
loadPrcFileData("", "framebuffer-multisample 1")
loadPrcFileData("", "multisamples 4")

class MyApp(ShowBase):

    def __init__(self):
        ShowBase.__init__(self)

        window = WindowProperties()
        window.fixed_size = True
        window.setSize(1024, 1024) #dimensioni ben definite rendono più semplice la simulazione e i calcoli che ne derivano
        self.win.requestProperties(window)
        print(cv2.__version__)

        self.image1 = PNMImage()
        self.image2 = PNMImage()

        self.scene = self.loader.loadModel("rio_garigliano.glb")
        self.scene.reparentTo(self.render)
        self.scene.setHpr(0, 90, 0)
        self.scene.setScale(1, 1, 1)
        self.scene.setPos(-8, 42, 0)

        ambLight = AmbientLight("ambient")
        ambLight.setColor(Vec4(0.6, 0.6, 0.6, 1.0))
        ambNode = self.render.attachNewNode(ambLight)
        self.render.setLight(ambNode)

        dirLight = DirectionalLight("directional")
        dirLight.setColor(Vec4(0.5, 0.6, 0.7, 1.0))
        dirNode = self.render.attachNewNode(dirLight)
        dirNode.setHpr(60, 0, 90)
        self.render.setLight(dirNode)

        sptLight = Spotlight("spot")
        sptLens = PerspectiveLens()
        sptLight.setLens(sptLens)
        sptLight.setColor(Vec4(0.8, 0.7, 0.5, 1.0))
        sptLight.setShadowCaster(True)
        sptNode = self.render.attachNewNode(sptLight)
        sptNode.setPos(-10, -10, 100)
        sptNode.lookAt(self.scene)
        self.render.setLight(sptNode)

        lensp = PerspectiveLens()
        lensp.setAspectRatio(1024/512)
        lensp.setFov(50)
        self.new_cam_np_1 = self.makeCamera(self.win, aspectRatio=2, lens=lensp, stereo=False, displayRegion=(0, 1, 0.5, 1))
        self.new_cam_np_2 = self.makeCamera(self.win, aspectRatio=2, lens=lensp, stereo=False, displayRegion=(0, 1, 0, 0.5))

        pos_cam_1 = -1
        pos_cam_2 = 1
        self.height = 50
        baseline = abs(pos_cam_1-pos_cam_2)

        self.new_cam_np_1.setPos(pos_cam_1, -42, self.height)
        self.new_cam_np_1.lookAt(pos_cam_1, -42, -2000)
        self.new_cam_np_2.setPos(pos_cam_2, -42, self.height)
        self.new_cam_np_2.lookAt(pos_cam_2, -42, -2000)

        self.dest_x = 0
        self.dest_y = 0
        self.dest_z = self.height

        self.counter = 10
        self.cycle = 0

        #contenitori delle bounding box + labels da visualizzare a schermo
        self.rectangles = []
        self.labels = []

        for dr in self.camNode.getDisplayRegions(): #è necessario spegnere la camera di default oppure 
            #la vista da quella camera si sovrappone a quella delle camere che ho creato
            dr.setActive(False)

        model = YOLO('C:/Users/simon/Desktop/collodel_machine_vision/best7.pt')

        #print(self.GraphicsOutput.getOneShot())

        self.taskMgr.setupTaskChain('drone_chain', numThreads = 0, tickClock = True, frameBudget = -1, frameSync = True, timeslicePriority = False)

        self.taskMgr.add(self.capture_screenshots_and_calculate_distance, "captureScreenshotsandUpdateVariables", taskChain="drone_chain", extraArgs=[lensp,baseline,model])


    def capture_screenshots_and_calculate_distance(self, lensp, baseline, model):

        #pulitura schermo
        for rectangle in self.rectangles:
            rectangle.destroy()
        self.rectangles.clear()

        for label in self.labels:
            label.destroy()
        self.labels.clear()

        self.graphicsEngine.syncFrame()
        self.graphicsEngine.renderFrame()

        
        pos_x, pos_y, pos_z = self.new_cam_np_1.getPos()
        
        #implementazione della finta navigazione del drone: si decide una coordinata a caso nello spazio e la si fa raggiungere al drone in dieci step
        if self.counter == 10:
            self.prev_x = self.dest_x
            self.prev_y = self.dest_y
            self.prev_z = self.dest_z 

            self.dest_x = random.randint(-80,30)
            self.dest_y = random.randint(-227,20)
            self.dest_z = random.randint(20,70)

            self.mov_x = (self.dest_x - self.prev_x)/10
            self.mov_y = (self.dest_y - self.prev_y)/10
            self.mov_z = (self.dest_z - self.prev_z)/10

            self.counter = 0

        dr1 = self.new_cam_np_1.node().getDisplayRegion(0)
        dr2 = self.new_cam_np_2.node().getDisplayRegion(0)

        #rendering immagini
        dr1.getScreenshot(self.image1)
        dr2.getScreenshot(self.image2)

        #salvataggio render (seh vabbé render è una parola granda)
        img_name = "camera1-"+str(pos_x)+"-"+str(pos_y)+".png"
        self.image1.write(Filename(img_name))
        self.image2.write(Filename("camera2-"+str(pos_x)+"-"+str(pos_y)+".png"))
        '''expfolder = 'exported'
        for oldphoto in os.listdir(expfolder):
            del_path = os.path.join(expfolder, oldphoto)
            os.unlink(del_path)
        self.image1.write(Filename("exported/f-"+str(pos_x)+"-"+str(pos_y)+".png"))'''


        #essendo l'unica camera disponibile su Panda3D la pinhole camera senza fare un procedimento ridicolo tramite il NonLinearImager faccio prima a distorcere l'immagine già renderizzata emulando una lente
        #add_distortion_example("C:/Users/simon/Downloads/camera1.png")
        #add_distortion_example("C:/Users/simon/Downloads/camera2.png")


        cv2_mat_1 = pnmimage_to_cv2mat(self.image1)
        cv2_mat_2 = pnmimage_to_cv2mat(self.image2)
        cv2_mat_1 = cv2.cvtColor(cv2_mat_1, cv2.COLOR_BGR2RGB) #altrimenti i colori sono in BGR perché boh opencv
        cv2_mat_2 = cv2.cvtColor(cv2_mat_2, cv2.COLOR_BGR2RGB)
        h1, w1 = cv2_mat_1.shape[:2]

        #essenzialmente le matrici che servono per la rectification (descrivono come sono i render)
        intrinsic_1,fx = get_intrinsic_matrix(lensp, h1, w1)


        #CODICE PER LA RETTIFICAZIONE: NON NECESSARIO CON PINHOLE CAMERA MA SI PUO' AGGIUNGERE UNA FINTA DISTORSIONE PER SIMULARE IL COMPORTAMENTO DI UNA LENTE
        '''intrinsic_2,fx_2 = get_intrinsic_matrix(lensp, h2, w2)
        intrinsic_1 = intrinsic_1.astype(np.float64) #cv2 costringe a usare float64 o non funziona 
        intrinsic_2 = intrinsic_2.astype(np.float64)
        #print(intrinsic_1)
        #print("width and height: ",w1,h1)
        distcoeffs1 = np.zeros(5, dtype=np.float64) #0 con la pinhole camera virtuale, altrimenti per altri tipi di lente va calcolata a parte
        distcoeffs2 = np.zeros(5, dtype=np.float64)

        T = np.array([2.,0.,0.], dtype=np.float64) #variabile che contiene la distanza tra le camere (varia in base all'asse)
        R = np.eye(3,dtype=np.float64)

        #stereo rectification usando la funzione di OpenCV
        R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
        intrinsic_1, distcoeffs1, intrinsic_2, distcoeffs2,(w1,h1), R, T) #


        #calcolo rectification maps
        map1x, map1y = cv2.initUndistortRectifyMap(intrinsic_1, distcoeffs1, R1, P1, [w1,h1], cv2.CV_32FC1)
        map2x, map2y = cv2.initUndistortRectifyMap(intrinsic_2, distcoeffs2, R2, P2, [w1,h1], cv2.CV_32FC1)

        #applicazione della rectification map (in input prende le immagini esportate)
        rectified_img1 = cv2.remap(cv2_mat_1, map1x, map1y, cv2.INTER_LINEAR)
        rectified_img2 = cv2.remap(cv2_mat_2, map2x, map2y, cv2.INTER_LINEAR)

        cv2.imwrite("rectified-1-"+str(self.height)+".png", rectified_img1)
        cv2.imwrite("rectified-2-"+str(self.height)+".png", rectified_img2)

        #applica YOLO_V7 per calcolare la bounding box in una delle immagini'''

        if self.cycle%2 == 1: #i frame pari spostano la telecamera e salvano l'immagine, i frame dispari fanno l'analisi e il display delle bboxes
            prd, prd_results = predict(model, img_name)
        else:
            prd = False
        #print("Returned String: ",returned_string)

        if prd == True:
            bboxes = prd_results
            for k in range(len(bboxes)):
                bboxes[k][0] = bboxes[k][0] - bboxes[k][2]/2
                bboxes[k][1] = bboxes[k][1] - bboxes[k][3]/2  #ho configurato lo stereo matching e il display delle bounding box con in mente
                #che x e y fossero l'angolo in alto a sinistra, non il centro della bounding box
            t_bboxes = [[round(float(number)) for number in row] for row in bboxes] #trasformare i float in int
            bboxes = t_bboxes

            print("bboxes: ", bboxes)

            calculated_distances = calculate_disparity_and_distance(cv2_mat_1,cv2_mat_2,bboxes,w1,h1,baseline,fx,self.cycle)
            
            #codice per il disegno delle bounding box sullo schermo
            if self.cycle%2 == 1: #a causa dell'async del render questo è l'unico modo di vedere le bounding box per qualche ragione
                did = self.draw_bboxes(bboxes,calculated_distances,5)

            for i in range(len(calculated_distances)):
                print(f"\nDistance object {i}: {calculated_distances[i]}")
        else:
            print("Nothing found")
            
        #print("Camera height: ",pos_z)

        #vengono spostate le videocamere seguendo una traiettoria casuale
        if self.cycle%2 == 0: #a causa dell'async del render questo è l'unico modo di vedere le bounding box per qualche ragione
            self.new_cam_np_1.setPos(pos_x + self.mov_x, pos_y + self.mov_y, 25)
            self.new_cam_np_1.lookAt(pos_x + self.mov_x, pos_y + self.mov_y, -2000)
            self.new_cam_np_2.setPos(pos_x + self.mov_x + 2, pos_y + self.mov_y, 25)
            self.new_cam_np_2.lookAt(pos_x + self.mov_x + 2, pos_y + self.mov_y, -2000)
        self.counter += 1
        self.cycle +=1

        print("\n")
        return Task.cont
    
    
    def draw_bboxes(self,bboxes,calculated_distances,edge_width):

        print("initializing draw_bboxes")

        _, _, edge_width_n, edge_height_n = pixels_to_dframe(0, 0, edge_width, edge_width)

        #Per ogni bounding box vengono fatti diversi display di rettangoli. Questo perché Panda3D non permette di creare un rettangolo vuoto
        #per rappresentare una bounding box, quindi bisogna creare un rettangolo per ogni lato e fare una serie di trasposizioni dal sistema di coordinate
        #di un'immagine opencv (in pixel) in un sistema di Panda3D (ci sono due assi che vanno da -1 a 1, in più i rettangoli non sono disegnati partendo
        #da in alto a sinistra ma da in basso a sinistra)
        for bbox,distance in zip(bboxes,calculated_distances):

            #print(bbox)

            #Etichetta con l'altezza
            n_x, n_y, n_w, n_h = pixels_to_dframe(bbox[0], bbox[1], bbox[2], bbox[3])
            label = OnscreenText(text="Dist: "+str(distance), pos=(n_x, n_y), scale=0.04,
                                 fg=(1, 1, 1, 1), bg=(0, 0, 0, 0.5))
            
            self.labels.append(label)

            #rect0 = DirectFrame(frameColor=(0,0,1,0.5), frameSize=(0,n_w,0,-n_h), pos=((bbox[0]*2/1024)-1 ,0, 1-(bbox[1]/512)))
            #print("trasposed_coords = ",(bbox[0]*2/1024)-1,1-(bbox[1]/512))

            #Lato in alto
            rect1 = DirectFrame(frameColor=(1, 0, 0, 1), frameSize=(0, n_w, 0, edge_height_n), pos=(n_x, 0, n_y - edge_height_n))

            #Lato in basso
            n_x, n_y, n_w, _ = pixels_to_dframe(bbox[0], bbox[1] + bbox[3] - edge_width, bbox[2], edge_width)
            rect2 = DirectFrame(frameColor=(1, 0, 0, 1), frameSize=(0, n_w, 0, edge_height_n), pos=(n_x, 0, n_y - edge_height_n))

            #Lato a destra
            n_x, n_y, _, n_h = pixels_to_dframe(bbox[0] + bbox[2] - edge_width, bbox[1], edge_width, bbox[3] - 2 * edge_width)
            rect3 = DirectFrame(frameColor=(1, 0, 0, 1), frameSize=(0, edge_width_n, 0, n_h), pos=(n_x, 0, n_y - n_h - edge_height_n))
            
            #Lato a sinistra
            n_x, n_y, _, n_h = pixels_to_dframe(bbox[0], bbox[1], edge_width, bbox[3] - 2 * edge_width)
            rect4 = DirectFrame(frameColor=(1, 0, 0, 1), frameSize=(0, edge_width_n, 0, n_h), pos=(n_x, 0, n_y - n_h - edge_height_n))

            self.rectangles += [rect1, rect2, rect3, rect4] #si mettono in un array perché poi vanno eliminati a ogni nuovo frame

        self.graphicsEngine.syncFrame()
        self.graphicsEngine.renderFrame()
        return True



        


app = MyApp()
app.run()

