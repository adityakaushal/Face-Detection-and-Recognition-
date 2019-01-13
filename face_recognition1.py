'''
@author 
adityakaushal
'''
import pickle
import cv2
import os
from IPython.display import clear_output
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, KFold


def make_480p():
    cam.set(3, 640)
    cam.set(4, 480)


def detect_face(frame):
    detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = detector.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=3)
    return faces


def cut_faces(image, faces_coord):
    faces = []
    for (x, y, w, h) in faces_coord:
        #w_rm=int(0.2*w/2)
        faces.append(image[y:y+h, x:x+w])
    return faces


def normalize_intensity(images):
    images_norm = []
    for image in images:
        is_color = len(image.shape) == 3
        if is_color:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        images_norm.append(cv2.equalizeHist(image))
    return images_norm


def resize(images, size=(47, 62)):
    image_resize = []
    for image in images:
        if image.shape < size:
            img_size = cv2.resize(image, size, interpolation=cv2.INTER_CUBIC)
        else:
            img_size = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        image_resize.append(img_size)
    return image_resize


def normalize_faces(frame, faces_coord):
    faces = cut_faces(frame, faces_coord)
    faces = normalize_intensity(faces)
    faces = resize(faces)
    return faces


def plt_show(image, title=''):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    plt.axis("off")
    plt.title(title)
    plt.imshow(image, cmap='Greys_r')
    plt.show()


def draw_rectangle(image, coords):
    for (x, y, w, h) in coords:
        #w_rm=int(0.2*w/2)
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)


cam = cv2.VideoCapture(0)

make_480p()
folder = "People/" + input("Person:").lower()
if not os.path.exists(folder):
    os.mkdir(folder)
    flag_start_capturing = False
    sample = 1
    cv2.namedWindow("Face", cv2.WINDOW_AUTOSIZE)

    while True:
        ret, frame = cam.read()
        faces_coord = detect_face(frame)
        frame = cv2.flip(frame, 1, 0)
        if len(faces_coord):
            faces = normalize_faces(frame, faces_coord)
            cv2.imwrite(folder+'/'+str(sample) + '.jpg', faces[0])
            plt_show(faces[0], "Image saved :" + str(sample))
            clear_output(wait=True)

        draw_rectangle(frame, faces_coord)
        cv2.imshow("Face", frame)
        keypress = cv2.waitKey(1)

        if keypress == ord('c'):

            if flag_start_capturing == False:
                flag_start_capturing = True
            else:
                flag_start_capturing = False
                sample = 0
        if flag_start_capturing == True:
            sample += 1
        if sample == 20:
            break
            #if sample > 15:
            #break
    cam.release()
    cv2.destroyAllWindows()
else:
    print("This name already exits.")


basepath = "lfw_home/lfw_funneled"
images = os.listdir(basepath)
print(len(images))
data = images[:210]

for i, folder in enumerate(data, start=1):
    files = os.listdir(basepath+'/'+folder)
    for k, img in enumerate(files, start=1):
        if img.endswith(".jpg"):
            frame = cv2.imread(basepath+'/'+folder+'/'+img, 0)

            face_coord = detect_face(frame)
            if len(face_coord):
                faces = cut_faces(frame, face_coord)
                faces = normalize_intensity(faces)
                faces = resize(faces)
                cv2.imwrite('People/unknown/'+str(i)+'.jpg', faces[0])
                break


def collect_dataset():
    images = []
    labels = []
    labels_dic = {}

    people = [person for person in os.listdir("People/")]

    for i, person in enumerate(people):
        labels_dic[i] = person
        for image in os.listdir("People/"+person):
            if image.endswith('.jpg'):
                images.append(cv2.imread("People/"+person+'/'+image, 0))
                labels.append(i)
    return (images, np.array(labels), labels_dic)


images, labels, labels_dic = collect_dataset()
print(len(images))
print(labels_dic)

X_train = np.asarray(images)
X_train.shape

train = X_train.reshape(len(X_train), -1)

train.shape

pca1 = PCA(n_components=100)
pca1.fit(train)

pca1 = PCA(n_components=.99)
new_train = pca1.fit_transform(train)

print(pca1.n_components_)

param_grid = {'C': [.0001, .001, .01, .1, 1, 10]}

kf = KFold(n_splits=5, shuffle=True)

gs_svc = GridSearchCV(SVC(kernel='linear', probability=True),
                    param_grid=param_grid, cv=kf, scoring='accuracy')

gs_svc.fit(new_train, labels)

print(gs_svc.best_score_)

print(gs_svc.best_params_)

svc1 = gs_svc.best_estimator_

filename = 'svc_face.pkl'
f = open(filename, 'wb')
pickle.dump(svc1, f)
f.close()

filename = 'svc_face.pkl'
svc1 = pickle.load(open(filename, 'rb'))

cam = cv2.VideoCapture(0)
make_480p()
font = cv2.FONT_HERSHEY_PLAIN
cv2.namedWindow("opencv_face", cv2.WINDOW_AUTOSIZE)

while True:
    ret, frame = cam.read()
    frame = cv2.flip(frame, 1, 0)
    faces_coord = detect_face(frame)  # detect more than one face
    if len(faces_coord):
        faces = normalize_faces(frame, faces_coord)

        for i, face in enumerate(faces):  # for each detected face

            test = pca1.transform(face.reshape(1, -1))
            prob = svc1.predict_proba(test)
            confidence = svc1.decision_function(test)
            print(confidence)
            print(prob)

            pred = svc1.predict(test)
            print(pred, pred[0])

            pred = labels_dic[pred[0]].capitalize()
            threshold = .50

            if confidence > .75:
                cv2.putText(
                    frame, 'Aditya Kaushal', (faces_coord[i][0], faces_coord[i][1] - 10), cv2.FONT_HERSHEY_PLAIN, 3, (66, 53, 243), 2)

            else:
                cv2.putText(frame, 'Intruder', (
                    faces_coord[i][0], faces_coord[i][1] - 10), cv2.FONT_HERSHEY_PLAIN, 2, (66, 53, 243), 2)

    clear_output(wait=True)
    draw_rectangle(frame, faces_coord)  # rectangle around face
    cv2.putText(frame, "Press ESC to Exit", (5,
                                            frame.shape[0] - 5), cv2.FONT_HERSHEY_PLAIN, 1.3, (66, 53, 243), 2, cv2.LINE_AA)

    cv2.imshow("opencv_face", frame)  # live feed in external
    if cv2.waitKey(5) == 27:
        break

cam.release()
cv2.destroyAllWindows()
