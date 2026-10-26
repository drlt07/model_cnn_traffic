import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image

import numpy
# Tải dữ liệu đã được training
from tensorflow.keras.models import load_model

model = load_model('traffic_classifier.h5')

# Các nhãn của các biển đã training
classes = {0: 'Speed limit (20km/h)',
           1: 'Speed limit (30km/h)',
           2: 'Speed limit (50km/h)',
           3: 'Speed limit (60km/h)',
           4: 'Speed limit (70km/h)',
           5: 'Speed limit (80km/h)',
           6: 'End of speed limit (80km/h)',
           7: 'Speed limit (100km/h)',
           8: 'Speed limit (120km/h)',
           9: 'No passing',
           10: 'No passing for vehicles over 3.5 metric tons',
           11: 'Right-of-way at the next intersection',
           12: 'Priority road',
           13: 'Yield',
           14: 'Stop',
           15: 'No vehicles',
           16: 'Vehicles over 3.5 metric tons prohibited',
           17: 'No entry',
           18: 'General caution',
           19: 'Dangerous curve to the left',
           20: 'Dangerous curve to the right',
           21: 'Double curve',
           22: 'Bumpy road',
           23: 'Slippery road',
           24: 'Road narrows on the right',
           25: 'Road work',
           26: 'Traffic signals',
           27: 'Pedestrians',
           28: 'Children crossing',
           29: 'Bicycles crossing',
           30: 'Beware of ice/snow',
           31: 'Wild animals crossing',
           32: 'End of all speed and passing limits',
           33: 'Turn right ahead',
           34: 'Turn left ahead',
           35: 'Ahead only',
           36: 'Go straight or right',
           37: 'Go straight or left',
           38: 'Keep right',
           39: 'Keep left',
           40: 'Roundabout mandatory',
           41: 'End of no passing',
           42: 'End of no passing by vehicles over 3.5 metric tons'}

# Tên tiếng Việt tương ứng 43 nhãn
ten_viet = ['Tốc độ tối đa 20 km/h', 'Tốc độ tối đa 30 km/h', 'Tốc độ tối đa 50 km/h', 'Tốc độ tối đa 60 km/h', 'Tốc độ tối đa 70 km/h', 'Tốc độ tối đa 80 km/h', 'Hết hạn chế tốc độ 80 km/h', 'Tốc độ tối đa 100 km/h', 'Tốc độ tối đa 120 km/h', 'Cấm vượt', 'Cấm xe ô tô tải vượt', 'Giao nhau với đường không ưu tiên', 'Bắt đầu đường ưu tiên', 'Giao nhau với đường ưu tiên', 'Dừng lại', 'Đường cấm', 'Cấm xe ô tô tải', 'Cấm đi ngược chiều', 'Nguy hiểm khác', 'Chỗ ngoặt nguy hiểm vòng bên trái', 'Chỗ ngoặt nguy hiểm vòng bên phải', 'Nhiều chỗ ngoặt nguy hiểm liên tiếp', 'Đường có ổ gà, lồi lõm', 'Đường trơn', 'Đường bị hẹp về phía phải', 'Công trường', 'Giao nhau có tín hiệu đèn', 'Đường người đi bộ cắt ngang', 'Trẻ em', 'Đường người đi xe đạp cắt ngang', 'Đường có băng tuyết', 'Thú rừng vượt qua đường', 'Hết tất cả các lệnh cấm', 'Rẽ phải phía trước', 'Rẽ trái phía trước', 'Đi thẳng', 'Đi thẳng hoặc rẽ phải', 'Đi thẳng hoặc rẽ trái', 'Hướng phải đi vòng sang phải', 'Hướng phải đi vòng sang trái', 'Nơi giao nhau chạy theo vòng xuyến', 'Hết cấm vượt', 'Hết cấm xe ô tô tải vượt']

# Nếu có signs.json (lấy từ app) thì dùng tên biển trong đó, gồm cả các biển Việt Nam thêm vào
import os, json
if os.path.exists('signs.json'):
    ten_viet = [s['name'] for s in json.load(open('signs.json', encoding='utf-8'))]

# Tạo giao diện
top = tk.Tk()
top.geometry('1000x700')
top.title('Nhận diện biển báo giao thông')
top.configure(background='#00CC99')

label = Label(top, background='#006666', font=('arial', 15, 'bold'))
sign_image = Label(top)

def classify(file_path):
    global label_packed
    image = Image.open(file_path).convert('RGB')
    image = image.resize((30, 30))
    image = numpy.expand_dims(image, axis=0)
    image = numpy.array(image)
    print(image.shape)
    pred = model.predict(image)[0]
    print(max(pred))
    print(pred)
    best = int(numpy.argmax(pred))
    sign = classes.get(best, ten_viet[best])
    ten_vn = ten_viet[best]
    print(sign)
    if pred[best] < 0.6:
        label.configure(foreground='#B00020', text=f'Chưa chắc chắn ({pred[best]*100:.1f}%) – có thể là: {ten_vn}')
    else:
        label.configure(foreground='#011638', text=f'{ten_vn}\n({sign}) – độ tin cậy {pred[best]*100:.1f}%')


def show_classify_button(file_path):
    classify_b = Button(top, text="Nhận diện", command=lambda: classify(file_path), padx=10, pady=5)
    classify_b.configure(background='dark red', foreground='white', font=('arial', 10, 'bold'))
    classify_b.place(relx=0.79, rely=0.85)


def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width() / 2.25), (top.winfo_height() / 2.25)))
        im = ImageTk.PhotoImage(uploaded)

        sign_image.configure(image=im)
        sign_image.image = im
        label.configure(text='')
        show_classify_button(file_path)
    except:
        pass

upload = Button(top, text="Tải ảnh lên", command=upload_image, padx=10, pady=5)
upload.configure(background='#364156', foreground='white', font=('arial', 10, 'bold'))

upload.pack(side=BOTTOM, pady=50)
sign_image.pack(side=BOTTOM, expand=True)
label.pack(side=BOTTOM, expand=True)
heading = Label(top, text="Nhận diện biển báo giao thông bằng mạng CNN", pady=20, font=('arial', 20, 'bold'))

heading.configure(background='#CDCDCD', foreground='#364156')
heading.pack()
top.mainloop()
