import os
import cv2
import time
import tensornets as nets
import tensorflow as tf
import random
from skimage import color, transform
from keras.models import Model
from keras.layers import Input, LSTM
from keras.layers import Dense, TimeDistributed
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
import numpy as np


def model(clips, labels):
    print(np.shape(clips))
    print(np.shape(labels))

    clip_train, clip_test, label_train, label_test = train_test_split(clips, labels, test_size=0.3, random_state=130)
    clip_train = np.array(clip_train)
    label_train = np.array(label_train)
    batch = 15
    target = 2
    epochs = 30

    row_hidden = 128
    col_hidden = 128

    frame, row, col = (50, 144, 256)

    x = Input(shape=(frame, row, col))

    encoded_rows = TimeDistributed(LSTM(row_hidden))(x)
    encoded_columns = LSTM(col_hidden)(encoded_rows)
    prediction = Dense(target, activation='softmax')(encoded_columns)
    model = Model(x, prediction)
    model.compile(loss='categorical_crossentropy', optimizer='NAdam', metrics=['accuracy'])

    i = 0
    filepath = 'models.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False, mode='max')
    callbacks_list = [checkpoint]

    for i in range(epochs):
        c = list(zip(clip_train, label_train))
        random.shuffle(c)
        clip, label = zip(*c)

        x_batch = np.asarray([clip[i:i + batch] for i in range(0, len(clip), batch)])
        y_batch = np.array([label[i:i + batch] for i in range(0, len(label), batch)])

        for j, xb in enumerate(x_batch):

            model.fit(xb, y_batch[j],  ### fit training data
                      batch_size=len(xb),  ### reiterate batch size - in this case we already set up the batches
                      epochs=1,  ### number of times to run through each batch
                      validation_data=(clip_train, label_train),  ### validation set from up earlier in notebook
                      callbacks=callbacks_list)  ### save if better than previous!
    # saving the model
    model.save('finalmodel.h5')

    scores = model.evaluate(clip_test, label_test, verbose=0)  ### score model
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])


def down_scale(image):

    cv2.imwrite('color_img.jpg', image)
    img = cv2.imread('color_img.jpg')
    img = color.rgb2gray(img)
    tmp = transform.resize(img, (144, 256))
    return tmp


def annotation(path, filename):
    input_m = tf.compat.v1.placeholder(tf.float32, [None, 416, 416, 3])
    model = nets.YOLOv3COCO(input_m, nets.Darknet19)

    classes = {'0': 'person', '2': 'car'}
    list_of_classes = [0, 2]
    with tf.compat.v1.Session() as sess:
        # with tf.Session() as sess:
        sess.run(model.pretrained())
        # "D://pyworks//yolo//videoplayback.mp4"
        cap = cv2.VideoCapture(os.path.join(path, filename))
        video_name = os.path.join(path, "updated"+filename)
        images = []

        while (cap.isOpened()):
            ret, frame = cap.read()
            img = cv2.resize(frame, (416, 416))
            imge = np.array(img).reshape(-1, 416, 416, 3)
            start_time = time.time()
            preds = sess.run(model.preds, {input_m: model.preprocess(imge)})

            print("--- %s seconds ---" % (time.time() - start_time))
            boxes = model.get_boxes(preds, imge.shape[1:3])
            cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            # height, width, layers = frame.shape

            cv2.resizeWindow('image', 700, 700)
            # print("--- %s seconds ---" % (time.time() - start_time))
            boxes1 = np.array(boxes)
            for j in list_of_classes:
                count = 0
                if str(j) in classes:
                    lab = classes[str(j)]
                if len(boxes1) != 0:

                    for i in range(len(boxes1[j])):
                        box = boxes1[j][i]

                        if boxes1[j][i][4] >= .40:
                            count += 1

                            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 1)
                            cv2.putText(img, lab, (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255),
                                        lineType=cv2.LINE_AA)
                print(lab, ": ", count)

            # cv2.imshow("image", img)
            images.append(img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        height, width, layers = images[0].shape

        video = cv2.VideoWriter(video_name, 0, 1, (width, height))

        for image in images:
            video.write(cv2.imread(os.path.join(path, image)))

    cap.release()
    cv2.destroyAllWindows()
    video.release()

def label_matrix(values):
    n_values = np.max(values) + 1
    return np.eye(n_values)[values]


# def create_matrix(label):
#     return np.eye(2)[label]

def load_annotated_data(path, file_type):
    files = sorted(os.listdir(path))

    labels = []
    clips = []

    if file_type == "pos":
        label = 1
    else:
        label = 0

    for filename in files:
        print(filename)
        frames = []
        # print(filename.startswith("annotated"))
        if filename.startswith("updated") and filename.endswith(".mp4"):
            cap = cv2.VideoCapture(os.path.join(path, filename))

            for i in range(50):
                print(len(frames))
                ret, frame = cap.read()
                frames.append(down_scale(frame))

            cap.release()
            cv2.destroyAllWindows()
            labels.append(label)
            clips.append(frames)

    return clips, labels


def load_data(path):
    files = sorted(os.listdir(path))
    for filename in files:
        print(filename)
        if filename.endswith(".mp4"):
            annotation(path, filename)

        return path


# #positive files
# #negative files
img_filepath = '<Enter Path>'  #### the filepath for the training video set

negative = img_filepath + "<Enter Folder name for negative>"
positive = img_filepath + "<>Enter Folder name for positive>"

data = []
load_data(negative)

# label = []
dt, neg_label = load_annotated_data(negative, "neg")
print(neg_label)
data.append(dt)

load_data(positive)
dt, pos_label = load_annotated_data(positive, "pos")
print(pos_label)
data.append(dt)

labels = np.concatenate((pos_label, neg_label))
labels = label_matrix(labels)

model(data, labels)
# path_annotated = load_data(neg)
# /Users/sarthakgupte/PycharmProjects/CV/Sarthak_Gupte_HW3.py

