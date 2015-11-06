#!/usr/bin/env python
from PySide import QtCore, QtGui

class ParameterWidget(QtGui.QWidget):
    def __init__(self, name, value):
        super(ParameterWidget, self).__init__()
        self.initUI(name, value)

    def initUI(self, name, value):
        self.main_box = QtGui.QHBoxLayout()

        self.sld = QtGui.QSlider(QtCore.Qt.Vertical)
        #sld.setFocusPolicy(QtCore.Qt.NoFocus)
        self.sld.setGeometry(30, 40, 100, 30)
        self.sld.valueChanged[int].connect(self.slider_update)

        label = QtGui.QLabel()
        label.setText(name)

        self.spinbox = QtGui.QDoubleSpinBox()
        self.spinbox.valueChanged.connect(self.spinbox_update)

        self.main_box.addWidget(label)
        self.main_box.addStretch(1)
        self.main_box.addWidget(self.spinbox)
        self.main_box.addStretch(1)
        self.main_box.addWidget(self.sld)
        self.main_box.addStretch(1)
        self.setLayout(self.main_box)

    def slider_update(self, value):
        self.pval = value
        self.spinbox.setValue(value)

    def spinbox_update(self, value):
        self.pval = value
        self.sld.setValue(value)
