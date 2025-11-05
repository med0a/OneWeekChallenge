import cv2
import face_recognition

# upload the picture
img=cv2.imread("./images/ronaldo.webp")
# change the color of the picture -> to be usable by the face_recognition library
rgb_img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
# we encode the image on a vector of 128 characters.
img_encoding=face_recognition.face_encodings(rgb_img)[0]

# doing the same thing with another picture 
# every image will have a different codage

img2=cv2.imread("./images/cilian.jpg")
rgb_img2=cv2.cvtColor(img2,cv2.COLOR_BGRA2BGR)
img2_encoding=face_recognition.face_encodings(rgb_img2)[0]
# comparing the two pictures
result=face_recognition.compare_faces([img_encoding],img2_encoding)
print("result", result)
# display the picture
cv2.imshow("img",img)
cv2.imshow("img2",img2)
# wait until a key is pressed to show the image
cv2.waitKey(0) 
