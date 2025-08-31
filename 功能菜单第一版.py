import torch
import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QMenu, QPushButton, QFileDialog, QListWidget, QMessageBox, \
    QAbstractItemView, QListWidgetItem, QHBoxLayout,QVBoxLayout,QLabel,QWidget
from PyQt6.QtGui import QIcon, QAction,QPixmap
from PyQt6.QtCore import Qt
#from PyQt6.QtWidgets.QMainWindow import centralWidget
from matplotlib import pyplot as plt
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
from PIL import ImageDraw,ImageFont
import time
import os
import mmcv
import cv2
import shutil
import numpy as np
#font = ImageFont.truetype('SimHei.ttf', 32)

class Menu(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.idx_to_labels={
            0: 'Hami Melon',
            1: 'Cherry Tomatoes',
            2: 'Mangosteen',
            3: 'Red Bayberry',
            4: 'Pomelo',
            5: 'Lemon',
            6: 'Longan',
            7: 'Pear',
            8: 'Coconut',
            9: 'Durian',
            10: 'Pitaya',
            11: 'Kiwifruit',
            12: 'Pomegranate',
            13: 'Tangerine',
            14: 'Carrot',
            15: 'Orange',
            16: 'Mango',
            17: 'Balsam Pear',
            18: 'Red Apple',
            19: 'Green Apple',
            20: 'Strawberry',
            21: 'Litchi',
            22: 'Pineapple',
            23: 'White Grape',
            24: 'Red Grape',
            25: 'Watermelon',
            26: 'Tomato',
            27: 'Cherry',
            28: 'Banana',
            29: 'Qucumber'
        }






    def load_model_from_file(self):         #导入模型文件
        file, _ = QFileDialog.getOpenFileName(self, "选择文件", "", "ckpt Files (*.pth)")
        if file:
            if file not in self.file_list:#新的模型，加入列表
                print(f"选中的文件: {file}")
                self.file_list.append(file)
                self.list_widget.addItem(file)
            else:       #已经加入过了，提示
                QMessageBox.information(self, "提示", "该文件已在列表中。")



    def load_image_from_file(self):             #导入图片
        file, _ = QFileDialog.getOpenFileName(self, "选择文件", "", "Media Files (*.png *.jpeg *.mp4 *.jpg)")
        if file:
            if file not in self.media_list:
                print(f"选中的文件: {file}")
                self.media_list.append(file)
                self.list_image_widget.addItem(file)
            else:
                QMessageBox.information(self, "提示", "该image已在列表中。")
            print(f"选中的image: {file}")


    def delete_from_media_list(self):
        selected_images = self.list_image_widget.selectedItems()
        if not selected_images:
            QMessageBox.information(self,'warning','no selected yet')
        for item in selected_images: #循环便利选中的模型依次移除列表
            self.list_image_widget.takeItem(self.list_image_widget.row(item))
            file_path = item.text()
            if file_path in self.media_list:
                self.media_list.remove(file_path)
        print(f"删除后文件列表: {self.file_list}")



    def delete_from_list(self):         #从列表中删除选中的模型
        selected_items = self.list_widget.selectedItems()
        if not selected_items:      #未选中，警告
            QMessageBox.information(self,"warning","no selected yet")
        for item in selected_items: #循环便利选中的模型依次移除列表
            self.list_widget.takeItem(self.list_widget.row(item))
            file_path = item.text()
            if file_path in self.file_list:
                self.file_list.remove(file_path)
        print(f"删除后文件列表: {self.file_list}")


    def get_test_transform(self,resize=256, crop_size=224,
                           mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225]):
        test_transform = transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        return test_transform


    def on_click_show_image(self):
        selected_images = self.list_image_widget.selectedItems()
        if not selected_images:
            QMessageBox.information(self,'warning','no selected yet')
            return
        self.image_filename = str(selected_images[0].text())
        self.imageDis1.setPixmap(QPixmap(self.image_filename).scaled(450, 300))




    def inference(self):#推理模型
        media_type = ['png','jpeg','jpg']
        selected_items = self.list_widget.selectedItems()
        selected_images = self.list_image_widget.selectedItems()
        if not selected_items or not selected_images:
            QMessageBox.information(self, "warning", "no selected yet")
            return

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        model_path = selected_items[0].text()
        print(f"加载模型: {model_path}")

        model = torch.load(model_path)
        model = model.eval().to(device)

        print(selected_images)
        img_path = str(selected_images[0].text())
        print(img_path)
        #img_pil = Image.open(img_path)
        img_pil = Image.open(img_path).convert('RGB')

        input_video = str(selected_images)
        output_path = '.\\output_pred.mp4'

        fileType = str(img_path).split('.')
        print(fileType)
        print(len(fileType)-1)
        if fileType[len(fileType)-1] not in media_type: # 创建临时文件夹，存放每帧结果
            temp_out_dir = time.strftime('%Y%m%d%H%M%S')
            os.mkdir(temp_out_dir)
            print('创建临时文件夹 {} 用于存放每帧预测结果'.format(temp_out_dir))
            # 读入待预测视频
            imgs = mmcv.VideoReader(input_video)

            prog_bar = mmcv.ProgressBar(len(imgs))

            # 对视频逐帧处理
            for frame_id, img in enumerate(imgs):
                ## 处理单帧画面
                img, pred_softmax = self.pred_single_frame(self.idx_to_labels, model, device, img, 'mp4')

                # 将处理后的该帧画面图像文件，保存至 /tmp 目录下
                cv2.imwrite(f'{temp_out_dir}/{frame_id:06d}.jpg', img)

                prog_bar.update()  # 更新进度条

            # 把每一帧串成视频文件
            mmcv.frames2video(temp_out_dir, output_path, fps=imgs.fps, fourcc='mp4v')

            shutil.rmtree(temp_out_dir)  # 删除存放每帧画面的临时文件夹
            print('删除临时文件夹', temp_out_dir)
            print('视频已生成', output_path)
        else:
            img_bgr, pred_softmax = self.pred_single_frame(self.idx_to_labels, model, device, img_pil,
                                                           os.path.splitext(img_path)[1])

            fig = plt.figure(figsize=(18, 6))
            # 绘制左图-预测图
            ax1 = plt.subplot(1, 2, 1)
            ax1.imshow(img_bgr)
            ax1.axis('off')
            # 绘制右图-柱状图
            ax2 = plt.subplot(1, 2, 2)
            x = list(self.idx_to_labels.values())
            y = pred_softmax.cpu().detach().numpy()[0] * 100
            ax = plt.bar(x, y, 0.45)
            ax2.bar(x, y, alpha=0.5, width=0.3, color='yellow', edgecolor='red', lw=3)
            plt.bar_label(ax, fmt='%.2f', fontsize=10)  # 置信度数值

            plt.title('{} results'.format(img_path), fontsize=30)
            plt.xlabel('label', fontsize=20)
            plt.ylabel('confidence', fontsize=20)
            plt.ylim([0, 110])  # y轴取值范围
            ax2.tick_params(labelsize=16)  # 坐标文字大小
            plt.xticks(rotation=90)  # 横轴文字旋转

            plt.tight_layout()
            fig.savefig('output/prediction+bar_chart.jpg')
            self.imageDis2.setPixmap(QPixmap('output/prediction+bar_chart.jpg').scaled(450, 300))


            
    def pred_single_frame(self,idx_to_labels,model,device,img_pil,file_type, n=5):
        print(file_type)
        '''
        输入摄像头画面bgr-array，输出前n个图像分类预测结果的图像bgr-array
        '''
        if file_type == 'mp4':
            test_transform = self.get_test_transform()
            img_rgb = cv2.cvtColor(img_pil, cv2.COLOR_BGR2RGB)  # BGR 转 RGB
            img_pil = Image.fromarray(img_rgb)  # array 转 pil
            input_img = test_transform(img_pil).unsqueeze(0).to(device)  # 预处理
            pred_logits = model(input_img)  # 执行前向预测，得到所有类别的 logit 预测分数
            pred_softmax = F.softmax(pred_logits, dim=1)  # 对 logit 分数做 softmax 运算
            print(pred_softmax)

            top_n = torch.topk(pred_softmax, n)  # 取置信度最大的 n 个结果
            pred_ids = top_n[1].cpu().detach().numpy().squeeze()  # 解析出类别
            confs = top_n[0].cpu().detach().numpy().squeeze()  # 解析出置信度

            # 在图像上写字
            draw = ImageDraw.Draw(img_pil)
            # 在图像上写字
            for i in range(len(confs)):
                pred_class = idx_to_labels[pred_ids[i]]
                text = '{:<15} {:>.3f}'.format(pred_class, confs[i])
                # 文字坐标，中文字符串，字体，rgba颜色
                draw.text((50, 100 + 50 * i), text,fill=(255, 0, 0, 1))

            img_pil = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)  # RGB转BGR

        else:
            test_transform = self.get_test_transform()
            print(test_transform)
            input_img = test_transform(img_pil)# 预处理
            print(input_img)
            input_img = input_img.unsqueeze(0).to(device)
            print(input_img)
            # 执行前向预测，得到所有类别的 logit 预测分数
            pred_logits = model(input_img)
            print(pred_logits)
            pred_softmax = F.softmax(pred_logits, dim=1)  # 对 logit 分数做 softmax 运算\
            print(pred_softmax)
        return img_pil, pred_softmax








    def initUI(self):

        self.setGeometry(100, 100, 600, 600)  # show整个界面
        self.setWindowTitle('演示菜单')

        main_layout = QVBoxLayout()
        btn_layout = QHBoxLayout()
        image_layout = QHBoxLayout()

        exitAct = QAction(QIcon('exit.png'),'&Exit',self)#退出按钮
        exitAct.setShortcut('Ctrl+Q')
        exitAct.setStatusTip('Exit APP')
        exitAct.triggered.connect(QApplication.instance().quit)

        self.statusBar()

        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&X')
        fileMenu.addAction(exitAct)

        sub1 = QMenu('Import', self)        #子菜单1
        sub2 = QMenu('portIm',self)         #子菜单2
        sub3 = QMenu('iorpmt',self)         #子菜单3

        fileMenu.addMenu(sub1)
        fileMenu.addMenu(sub2)
        fileMenu.addMenu(sub3)

        btn_load_model = QPushButton('LoadModel', self) #导入模型按钮
        #btn_load_model.move(30,20)
        btn_load_model.clicked.connect(self.load_model_from_file)

        btn_load_picture = QPushButton('LoadPicture', self) #导入图片按钮
        #btn_load_picture.move(150, 20)
        btn_load_picture.clicked.connect(self.load_image_from_file)

        self.file_list = []     #模型列表展示
        self.list_widget = QListWidget(self)
        self.list_widget.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        #self.list_widget.setGeometry(10, 50, 510, 200)

        self.media_list = []  # 模型列表展示
        self.list_image_widget = QListWidget(self)
        self.list_image_widget.itemClicked.connect(self.on_click_show_image)

        self.list_image_widget.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        #self.list_image_widget.setGeometry(10, 250, 510, 850)

        self.delete_button = QPushButton("删除模型选中项", self) #删除模型按钮
        #self.delete_button.move(270,20)
        self.delete_button.clicked.connect(self.delete_from_list)

        self.delete_media_button= QPushButton("删除媒体选中项", self)  # 删除模型按钮
        #self.delete_media_button.move(500, 20)
        self.delete_media_button.clicked.connect(self.delete_from_media_list)


        self.inference_btn = QPushButton('推理',self) #推理按钮
        #self.inference_btn.move(390,20)
        self.inference_btn.clicked.connect(self.inference)

        self.imageDis1 = QLabel()
        self.imageDis2 = QLabel()


        btn_layout.addWidget(btn_load_model)
        btn_layout.addWidget(btn_load_picture)
        btn_layout.addWidget(self.delete_button)
        btn_layout.addWidget(self.inference_btn)
        btn_layout.addWidget(self.delete_media_button)

        image_layout.addWidget(self.list_image_widget)
        image_layout.addWidget(self.imageDis1)
        image_layout.addWidget(self.imageDis2)

        main_layout.addLayout(btn_layout)
        main_layout.addLayout(image_layout)

        main_layout.addWidget(QLabel("模型列表:"))
        main_layout.addWidget(self.list_widget)

        main_layout.addWidget(QLabel("图片列表:"))
        main_layout.addWidget(self.list_image_widget)


        main_layout.setStretchFactor(self.list_widget, 1)
        main_layout.setStretchFactor(self.list_image_widget, 2)

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)



        self.show()






def main():                         #主程序
    app=QApplication(sys.argv)
    Me=Menu()
    sys.exit(app.exec())

main()




