#!/usr/bin/env python
import sys
import matplotlib
import numpy as np

matplotlib.use('Qt4Agg')
matplotlib.rcParams['backend.qt4']='PySide'
import pylab

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from PySide import QtCore, QtGui

class ParameterWidget(QtGui.QWidget):
    def __init__(self, name, value):
        super(ParameterWidget, self).__init__()
        self.initUI(name, value)

    def initUI(self, name, value):
        self.main_box = QtGui.QHBoxLayout()

        sld = QtGui.QSlider(QtCore.Qt.Horizontal, self)
        sld.valueChanged[int].connect(self.changeValue)

        label = QtGui.QLabel(self)
        label.setText(name)

        spinbox = QtGui.QDoubleSpinBox(self)

        self.main_box.addWidget(label)
        self.main_box.addStretch(1)
        self.main_box.addWidget(sld)
        self.main_box.addStretch(1)
        self.main_box.addWidget(spinbox)
        self.main_box.addStretch(1)

    def changeValue(self, value):
        print value

class Example(QtGui.QMainWindow):
    def __init__(self):
        super(Example, self).__init__()
        self.initUI()

    def initUI(self):
        self.main_frame = QtGui.QWidget()

        okButton = QtGui.QPushButton("OK")
        cancelButton = QtGui.QPushButton("Cancel")

        main_box = QtGui.QHBoxLayout()
        left_box = QtGui.QVBoxLayout()
        right_box = QtGui.QVBoxLayout()

        main_box.addLayout(left_box)
        main_box.addStretch(1)
        main_box.addLayout(right_box)
        left_box.addWidget(okButton)
        left_box.addWidget(cancelButton)


        fig = Figure(figsize=(600,600), dpi=72, facecolor=(1,1,1), edgecolor=(0,0,0))
        ax = fig.add_subplot(111)
        x = np.linspace(0, 2*np.pi)
        line, = ax.plot(x, np.sin(x))

        canvas = FigureCanvas(fig)
        mpl_toolbar = NavigationToolbar(canvas, self.main_frame)

        right_box.addWidget(canvas)
        right_box.addWidget(mpl_toolbar)

        p1 = ParameterWidget('a', 1.0)
        p2 = ParameterWidget('b', 1.0)

        left_box.addWidget(p1)
        left_box.addWidget(p2)

        self.main_frame.setLayout(main_box)
        self.setCentralWidget(self.main_frame)

        self.setGeometry(300, 300, 1200, 600)
        self.setWindowTitle('Buttons')
        self.show()

def main():
    app = QtGui.QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
