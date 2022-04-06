# -*- coding: utf-8 -*-
#
# Form implementation generated from reading ui file 'FormUI.ui'
#
# Created by: PyQt5 UI code generator 5.10.1
#
# WARNING! All changes made in this file will be lost!
 
from PyQt5 import QtCore, QtGui, QtWidgets,Qt
import cv2
from utils import *
import copy
class Ui_Form(object):  # 图形界面类
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1000,800)
        # 按钮
        self.openButton = QtWidgets.QPushButton(Form)
        self.openButton.setGeometry(QtCore.QRect(850, 50, 100, 28))
        self.openButton.setObjectName("openButton")
        
        # 文本框
        self.lineEdit = QtWidgets.QLineEdit(Form)
        self.lineEdit.setGeometry(QtCore.QRect(850, 100, 100, 50))  # x坐标,y坐标,宽,高
        self.lineEdit.setObjectName("lineEdit")
        # 按钮
        self.pushButton = QtWidgets.QPushButton(Form)
        self.pushButton.setGeometry(QtCore.QRect(850, 200, 100, 28))
        self.pushButton.setObjectName("pushButton")
        # 标签
        self.label = QtWidgets.QLabel(Form)
        self.label.setGeometry(QtCore.QRect(50, 50, 750, 750))
        self.label.setObjectName("label")
        # self.label.setScaledContents(True)


        self.score_label = QtWidgets.QLabel(Form)
        self.score_label.setGeometry(QtCore.QRect(850, 280, 100, 28))
        self.nms_label = QtWidgets.QLabel(Form)
        self.nms_label.setGeometry(QtCore.QRect(850, 330, 100, 28))
        self.mask_label = QtWidgets.QLabel(Form)
        self.mask_label.setGeometry(QtCore.QRect(850, 380, 100, 28))
        
        self.score_slider = QtWidgets.QDoubleSpinBox(Form)
        self.score_slider.setRange(0,1.0)
        self.score_slider.setSingleStep(0.02)
        self.score_slider.setGeometry(QtCore.QRect(850, 300, 100, 28))
        self.score_slider.setValue(0.2)
        
        self.nms_slider = QtWidgets.QDoubleSpinBox(Form)
        self.nms_slider.setRange(0,1.0)
        self.nms_slider.setSingleStep(0.02)
        self.nms_slider.setGeometry(QtCore.QRect(850, 350, 100, 28))
        self.nms_slider.setValue(0.5)

        self.mask_slider = QtWidgets.QDoubleSpinBox(Form)
        self.mask_slider.setRange(0,1.0)
        self.mask_slider.setSingleStep(0.02)
        self.mask_slider.setGeometry(QtCore.QRect(850, 400, 100, 28))
        self.mask_slider.setValue(0.3)


        
        self.retranslateUi(Form)

    def retranslateUi(self, Form):  # 设置各组件的文本
        _translate = QtCore.QCoreApplication.translate
        # Form.setWindowTitle(_translate("Form", "这是一个测试窗口"))
        self.pushButton.setText(_translate("Form", "检测"))
        self.openButton.setText(_translate("Form", "打开图片"))
        self.label.setText(_translate("Form", "图片"))
        self.score_label.setText(_translate("Form", "score_thresh"))
        self.nms_label.setText(_translate("Form", "nms_thresh"))
        self.mask_label.setText(_translate("Form", "mask_thresh"))

class MyWindow(QtWidgets.QWidget,Ui_Form):          
    def __init__(self, parent=None):
        super(MyWindow, self).__init__(parent)
        self.setupUi(self)                          
        QtWidgets.QApplication.setStyle(QtWidgets.QStyleFactory.create('Fusion'))   # 界面风格
        
        self.pushButton.clicked.connect(self.btn_clicked)   
        self.openButton.clicked.connect(self.lab_clicked)   
        self.score_slider.valueChanged.connect(self.update_thresh)   
        self.nms_slider.valueChanged.connect(self.update_thresh)   
        self.mask_slider.valueChanged.connect(self.update_thresh)   

        self.ori_img = None
        self.features = None
        self.emb = None
        self.vocabulary=[]
        self.score_thresh = 0.2
        self.nms_thresh =0.5
        self.mask_thresh =0.3

    def update_thresh(self):
        self.score_thresh = self.score_slider.value()
        self.nms_thresh = self.nms_slider.value()
        self.mask_thresh = self.mask_slider.value()
        self.update_img()


    def btn_clicked(self):    
        input_words = self.lineEdit.text() 
        self.vocabulary = [input_words]
        self.emb = text_encoder.get_features(self.vocabulary)
        self.update_img()
    
    def update_img(self):
        boxes,cls_feat_stage_1,cls_feat_stage_2,cls_feat_stage_3,mask_pred,proposal_scores =  copy.deepcopy(self.features)        
        postprocesser = PostProcess()
        scores = postprocesser.decode_scores(self.emb,[cls_feat_stage_1,cls_feat_stage_2,cls_feat_stage_3],[proposal_scores])
        score = scores[0]
        box = boxes
        mask_feat = mask_pred
        height, width = self.ori_img.shape[:2]
        ori_image_sizes=[height, width]
        print('===box',box,ori_image_sizes)
        class_id, box, score, image_mask = postprocesser.decode_box_mask(score
                                        ,box
                                        ,mask_feat
                                        ,self.score_thresh
                                        ,self.nms_thresh
                                        ,self.mask_thresh
                                        ,ori_image_sizes
                                        )
        print("====class_id",class_id,box)
        img = visualization(self.ori_img.copy(),self.vocabulary,class_id, box, score, image_mask)
        cv2.imwrite('res.jpg',img)
        qImg = QtGui.QImage(img.data,img.shape[1],img.shape[0],img.shape[1]*3,QtGui.QImage.Format_RGB888).rgbSwapped()
        qImg = qImg.scaledToWidth(750)
        self.label.setPixmap(QtGui.QPixmap.fromImage(qImg))    
    
    def lab_clicked(self):
        imgName, imgType = QtWidgets.QFileDialog.getOpenFileName(self, "打开图片", "", "*.jpg;;*.png;;All Files(*)")
        if imgName=='':
            return
        self.ori_img = cv2.imread(imgName)
        img=self.ori_img.copy()
        
        qImg = QtGui.QImage(img.data,img.shape[1],img.shape[0],img.shape[1]*3,QtGui.QImage.Format_RGB888).rgbSwapped()
        qImg = qImg.scaledToWidth(750)
        img_cv = self.ori_img.copy()
        # img_cv = ResizeShortestEdge(img_cv)
        img_cv = img_cv[:, :, ::-1] #RGB
        img_cv = preprocess(img_cv)
        self.features = mymodel.get_features(img_cv)
        self.label.setPixmap(QtGui.QPixmap.fromImage(qImg))







if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    myshow = MyWindow()
    myshow.show()
    sys.exit(app.exec_())