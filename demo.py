
# from ctpnport import *
from crnnport import *

#ctpn：Detecting Text in Natural Image with Connectionist Text Proposal Network
# text_detector = ctpnSource()

#crnn：识别Convolutional Recurrent Neural Network
model,converter = crnnSource()

timer = Timer()
print("\ninput exit break\n")

while 1 :
    im_name = raw_input("\nplease input file name:")
    if im_name == "exit":
       break
    im_path = "./img/" + im_name
    im = cv2.imread(im_path)
    if im is None:
      continue
    timer.tic()
    img,text_recs = getCharBlock(text_detector,im)
    crnnRec(model,converter,img,text_recs)
    print("Time: %f"%timer.toc())
    cv2.waitKey(0)