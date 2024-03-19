    #0. thêm thư viện
import cv2
from tensorflow.keras.applications import InceptionV3, MobileNet, VGG16, DenseNet121, EfficientNetB7
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Sequential, Model
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
import keras

#Số lớp
n_class = 9

#Hàm xây dụng mô hình
def get_model():
    #Tạo mô hình
    mobilenet = MobileNet(input_shape=(224, 224, 3), include_top=False, weights = 'imagenet')
    # Tạo mô hình chính
    model3 = mobilenet.output

    model3 = GlobalAveragePooling2D()(model3)
    model3 = Dense(1024, activation='relu')(model3)
    model3 = Dropout(0.25)(model3)
    model3 = Dense(1024, activation='relu')(model3)
    model3 = Dropout(0.25)(model3)
    model3 = Dense(512, activation='relu')(model3)
    outs = Dense( n_class, activation='softmax')(model3)
    
    #Khi ta train tất cả cá lớp của mobinet sẽ bị train lại. Vì thế ta nên đóng băng nó tại đây
    for layer in mobilenet.layers:
        layer.trainable = False

    model = Model(inputs = mobilenet.inputs, outputs= outs)

    return model

model = get_model()

    #3. make data
dataFolder = "Class"

#Đào tạo dữ liệu
train_datagen = ImageDataGenerator(preprocessing_function = keras.applications.mobilenet.preprocess_input, rotation_range=0,
                width_shift_range=0.2, height_shift_range=0.2, shear_range=0.3, zoom_range=0.5,  horizontal_flip=True, vertical_flip=True,
                validation_split=0.2 )

#flow_from_directory: Hàm này tạo ra một trình tạo dữ liệu từ thư mục chứa dữ liệu ảnh. Đỡ bị overfit
#DÙng để train
train_generator = train_datagen.flow_from_directory(
    dataFolder,                 #Thư mục huấn luyện
    target_size=(224,224),      #Xác định kích thước mục tiêu cho ảnh sau khi xử lý. Trong trường hợp này, ảnh sẽ được thu nhỏ về kích thước 224x224 pixel.
    batch_size=32,              #Xác định kích thước nhóm (batch size) cho mỗi lần lấy dữ liệu. Giá trị này ảnh hưởng đến hiệu suất và bộ nhớ khi huấn luyện mạng.
    class_mode='categorical',   #Xác định định dạng của nhãn (target). Trong trường hợp này, vì dữ liệu được tổ chức theo từng lớp, ta sử dụng 'categorical' để biểu diễn nhãn dưới dạng vectơ one-hot.
    subset='training')          #Xác định tập dữ liệu con (subset) mà hàm sẽ sử dụng. Trong trường hợp này, 'training' cho biết hàm sẽ chỉ lấy dữ liệu từ thư mục con 'training' bên trong dataFolder.

#Dùng để kiểm tra
validation_generator = train_datagen.flow_from_directory(
    dataFolder,
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical',
    subset='validation')        #Xác định tập dữ liệu con được sử dụng. Giá trị 'validation' cho biết trình tạo sẽ lấy dữ liệu từ tập dữ liệu xác thực (validation set).

classes = train_generator.class_indices #Lấy ra 1 từ điển từ thuộc tính class_indices
print(classes)
classes = list(classes.keys())  #Chuyển về dạng danh sách và chỉ lưu key
print(classes)

    #4.Train model
EPOCHS = 30
BS = 32

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

checkpoint = ModelCheckpoint('models/best.keras', monitor='val_loss', save_best_only = True, mode='auto')
callback_list = [checkpoint]        #Tạo 1 danh sách dựa trên đối tượng checkpoint

#Tính toán số bước (iterations) cần thiết cho mỗi epoch huấn luyện và xác thực
step_train = train_generator.n//BS      #  train_generator.n: Số lượng ảnh huẩn luyện
step_val = validation_generator.n//BS   # validation_generator.n: Số lượng ảnh từ tập dữ liệu xác thực

model.fit(x=train_generator, steps_per_epoch=step_train, validation_data= validation_generator,
                    validation_steps=step_val,
                    callbacks = callback_list,
                    epochs=EPOCHS)

    #5.Lưu model
model.save('models/modelMobilenet.h5')