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
'''
    - ImageDataGenerator: dùng để gia tăng (augmentation) dữ liệu ảnh cho quá trình huấn luyện mạng neural network.
        + Hàm này thực hiện chuẩn hóa đầu vào cho ảnh. Trong trường hợp này, nó sử dụng hàm preprocess_input được 
        thiết kế riêng cho kiến trúc mạng MobileNet của Keras. Hàm này thường sẽ trừ đi giá trị trung bình và 
        chia cho một hằng số để đưa dữ liệu ảnh về một khoảng giá trị cụ thể.
        + rotation_range=0: Tham số này xác định mức độ xoay ảnh theo góc (đơn vị độ). Giá trị 0 nghĩa là không xoay ảnh.
        + width_shift_range=0.2, height_shift_range=0.2: Xác định tỉ lệ dịch chuyển ảnh theo chiều ngang và dọc. 
        Ví dụ, width_shift_range=0.2 cho phép dịch chuyển ảnh tối đa 20% chiều rộng sang trái hoặc phải.
        + shear_range=0.3: Xác định mức độ cắt (shear) ảnh theo góc (radian). Giá trị 0.3 nghĩa là ảnh 
        có thể bị cắt nhẹ theo một hướng nhất định.
        + zoom_range=0.5: Xác định tỉ lệ zoom ảnh. Giá trị 0.5 cho phép phóng to hoặc thu nhỏ ảnh tối đa 50% so với kích thước gốc.
        + horizontal_flip=True, vertical_flip=True: Cho phép lật ngẫu nhiên ảnh theo chiều ngang và dọc, 
        giúp gia tăng dữ liệu và cải thiện khả năng khái quát của mạng.
        + validation_split=0.2: Dành ra 20% dữ liệu gốc để sử dụng cho quá trình đánh giá (validation) trong khi huấn luyện.

'''
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

'''
   - ModelCheckpoint dùng để lưu trữ mô hình mạng nơ-ron tốt nhất trong quá trình huấn luyện.Với các tham số sau:
    + Tham số đầu tiên là đường dẫn để lưu mô hình
    + monitor='val_loss': Xác định tiêu chí theo dõi để lưu trữ mô hình. Giá trị 'val_loss' cho biết mô hình sẽ được lưu khi giá trị val_loss 
    (mất mát xác thực) đạt giá trị tốt nhất.
    + save_best_only=True:Xác định chỉ lưu trữ mô hình khi nó đạt giá trị tốt nhất cho tiêu chí theo dõi. Giá trị True cho biết chỉ có mô hình 
    với val_loss tốt nhất mới được lưu.
    + mode='auto': Xác định phương thức so sánh giá trị theo dõi. Giá trị 'auto' tự động chọn phương thức so sánh dựa trên tiêu chí theo dõi 
    (val_loss là giá trị nhỏ hơn tốt hơn).
'''
checkpoint = ModelCheckpoint('models/best.keras', monitor='val_loss', save_best_only = True, mode='auto')
callback_list = [checkpoint]        #Tạo 1 danh sách dựa trên đối tượng checkpoint

#Tính toán số bước (iterations) cần thiết cho mỗi epoch huấn luyện và xác thực
step_train = train_generator.n//BS      #  train_generator.n: Số lượng ảnh huẩn luyện
step_val = validation_generator.n//BS   # validation_generator.n: Số lượng ảnh từ tập dữ liệu xác thực

'''
    - fit_generator: Đây là hàm dùng đẻ huấn luyện mô hinh
        + generator: Trình tạo dữ liệu train_generator cung cấp ảnh huấn luyện theo batch.
        + steps_per_epoch: Số bước thực hiện trên mỗi epoch huấn luyện (tính toán ở bước 2).
        + validation_data: Trình tạo dữ liệu validation_generator cung cấp ảnh xác thực theo batch.
        + validation_steps: Số bước thực hiện trên mỗi epoch xác thực (tính toán ở bước 2).
        + callbacks: Danh sách callbacks, trong trường hợp này là callback_list chứa checkpoint.
        + epochs: Số epoch (số lần lặp qua toàn bộ dữ liệu huấn luyện) cần thực hiện.
'''
model.fit(generator=train_generator, steps_per_epoch=step_train, validation_data= validation_generator,
                    validation_steps=step_val,
                    callbacks = callback_list,
                    epochs=EPOCHS)

    #5.Lưu model
model.save('models/modelMobilenet.h5')