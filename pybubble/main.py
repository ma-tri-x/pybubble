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
from parameter_widget import ParameterWidget

class MainApp(QtGui.QMainWindow):
    def __init__(self):
        super(MainApp, self).__init__()
        self.initUI()

    def initUI(self):
        self.main_frame = QtGui.QWidget()

        main_box = QtGui.QHBoxLayout()
        left_box = QtGui.QVBoxLayout()
        right_box = QtGui.QVBoxLayout()

        main_box.addLayout(left_box)
        main_box.addStretch(1)
        main_box.addLayout(right_box)

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
        self.setWindowTitle('Bubble-O Magic ODE Solver')

        exitAction = QtGui.QAction(QtGui.QIcon('exit.png'), '&Exit', self)
        exitAction.setShortcut('Ctrl+Q')
        exitAction.setStatusTip('Exit application')
        exitAction.triggered.connect(self.close)

        self.statusBar()
        menubar = self.menuBar()

        fileMenu  = menubar.addMenu('&File')
        fileMenu.addAction(exitAction)

        modelMenu = menubar.addMenu('&Model')
        modelMenu.addAction(exitAction)

        self.statusBar().showMessage('Ready')
        self.show()

def main():
    app = QtGui.QApplication(sys.argv)
    main_window = MainApp()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
