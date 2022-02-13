from PyQt5.QtWidgets import QApplication, QWidget, QLabel

# Only needed for access to command line arguments
import sys

# You need one (and only one) QApplication instance per application.
# Pass in sys.argv to allow command line arguments for your app.
# If you know you won't use command line arguments QApplication([]) works too.
app = QApplication(sys.argv)

# Create a Qt widget, which will be our window.
window = QWidget()
window.setWindowTitle('SWE Team3d')

#First two parameters are where the window will be placed on the screen.
#Second two parameters are the dimensions of the window.
window.setGeometry(100, 100, 280, 80)
window.move(60, 15)
helloMsg = QLabel('<h1>Testing!</h1>', parent=window)
helloMsg.move(60, 15)
#quitButton = QWidget.
window.show()


# Start the event loop.
sys.exit(app.exec_())


# Your application won't reach here until you exit and the event
# loop has stopped.
